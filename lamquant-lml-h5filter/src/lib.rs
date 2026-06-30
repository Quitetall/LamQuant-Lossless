//! LamQuant LML as an HDF5 (H5Z) **dynamically-loaded compression filter**.
//!
//! ADR 0051 Track 3, mechanism A — "own the NWB flank H.BWC is not addressing".
//! Built as a `cdylib`; drop the resulting `liblamquant_lml_h5filter.so` on
//! `HDF5_PLUGIN_PATH` and stock `h5py` / `pynwb` will transparently compress and
//! decompress integer datasets through the royalty-free LML lossless codec —
//! the file stays a fully native NWB/HDF5. `h5repack -f UD=32200 in.nwb out.nwb`
//! then losslessly shrinks an existing NWB *in place*, with HDF5 itself
//! preserving 100% of structure (groups, attributes, compound electrode tables,
//! object references) — only the integer dataset *chunks* are recoded.
//!
//! **Scope / safety guards.** The filter applies only to **little-endian
//! integer** datasets (`can_apply` refuses everything else, so float/compound
//! data is silently left to other filters — never corrupted). Each chunk is
//! self-describing (20-byte header), so decode never depends on `cd_values`
//! surviving. Big-endian storage is out of scope for now (x86/ARM-LE target).
//!
//! **Host-only.** Pulls in libhdf5 via `hdf5-metno-sys`; never compiled into the
//! no_std firmware floor.

mod hdf5_abi;
use hdf5_abi::{
    hid_t, H5Tget_class, H5Tget_order, H5Tget_sign, H5Tget_size, H5Zregister, H5Z_class2_t,
    H5Z_filter_t, H5allocate_memory, H5free_memory, H5Pget_chunk, H5Pmodify_filter,
    H5PL_TYPE_FILTER, H5T_INTEGER, H5T_ORDER_LE, H5T_SGN_2, H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE,
};
use std::os::raw::{c_int, c_uint, c_void};
use std::panic::catch_unwind;

use lamquant_lml_mcu::lml;

/// LamQuant LML filter ID. **Placeholder pending registration with The HDF
/// Group** (the 32xxx range is where third-party filters live; 32200 is unused
/// among the published assignments as of writing). Must match the `UD=` id used
/// by `h5py`/`h5repack`.
pub const LML_H5_FILTER_ID: H5Z_filter_t = 32200;

const FILTER_NAME: &[u8] = b"lamquant-lml\0";

// ── self-describing per-chunk header (20 bytes, little-endian) ──────────────
const MAGIC: [u8; 4] = *b"LMH1";
const HDR_LEN: usize = 20;
const METHOD_LML: u8 = 0; // body = lml::compress output
const METHOD_RAW: u8 = 1; // body = original bytes verbatim (LML didn't shrink)

#[inline]
fn read_int_le(b: &[u8], signed: bool) -> i64 {
    match (b.len(), signed) {
        (1, true) => b[0] as i8 as i64,
        (1, false) => b[0] as i64,
        (2, true) => i16::from_le_bytes([b[0], b[1]]) as i64,
        (2, false) => u16::from_le_bytes([b[0], b[1]]) as i64,
        (4, true) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        (4, false) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        (8, true) => i64::from_le_bytes(b[0..8].try_into().unwrap()),
        (8, false) => u64::from_le_bytes(b[0..8].try_into().unwrap()) as i64,
        _ => 0,
    }
}

/// Write the low `elem` bytes of `v` (two's-complement low bytes are identical
/// for signed/unsigned, so this is correct for both).
#[inline]
fn write_int_le(v: i64, elem: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes()[..elem]);
}

/// Compress: raw interleaved chunk bytes → header + LML body.
fn forward(input: &[u8], elem: usize, signed: bool, n_ch: usize) -> Option<Vec<u8>> {
    if elem == 0 || n_ch == 0 || input.len() % elem != 0 {
        return None;
    }
    let n_elem = input.len() / elem;
    // Row-major (T, C): linear index i = t*C + c ⇒ channel = i % C.
    let n_ch = if n_elem % n_ch == 0 { n_ch } else { 1 };
    let t = n_elem / n_ch;
    let mut channels: Vec<Vec<i64>> = (0..n_ch).map(|_| Vec::with_capacity(t)).collect();
    for i in 0..n_elem {
        let off = i * elem;
        channels[i % n_ch].push(read_int_le(&input[off..off + elem], signed));
    }

    let mut header = Vec::with_capacity(HDR_LEN);
    header.extend_from_slice(&MAGIC);
    header.push(METHOD_LML); // patched below if we fall back to raw
    header.push(elem as u8);
    header.push(signed as u8);
    header.push(0); // reserved
    header.extend_from_slice(&(n_ch as u32).to_le_bytes());
    header.extend_from_slice(&(input.len() as u64).to_le_bytes());
    debug_assert_eq!(header.len(), HDR_LEN);

    let body = lml::compress(&channels, 0).ok()?;
    // Never expand past raw: if LML didn't beat the raw bytes, store raw.
    if body.len() >= input.len() {
        header[4] = METHOD_RAW;
        let mut out = header;
        out.extend_from_slice(input);
        Some(out)
    } else {
        let mut out = header;
        out.extend_from_slice(&body);
        Some(out)
    }
}

/// Decompress: header + body → raw interleaved chunk bytes.
fn reverse(input: &[u8]) -> Option<Vec<u8>> {
    if input.len() < HDR_LEN || input[0..4] != MAGIC {
        return None;
    }
    let method = input[4];
    let elem = input[5] as usize;
    let _signed = input[6] != 0;
    let n_ch = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let orig_nbytes = u64::from_le_bytes(input[12..20].try_into().unwrap()) as usize;
    let body = &input[HDR_LEN..];

    if method == METHOD_RAW {
        return (body.len() == orig_nbytes).then(|| body.to_vec());
    }
    if elem == 0 || n_ch == 0 {
        return None;
    }
    let channels = lml::decompress(body).ok()?;
    if channels.len() != n_ch {
        return None;
    }
    let n_elem = orig_nbytes / elem;
    if n_elem % n_ch != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(orig_nbytes);
    for i in 0..n_elem {
        let c = i % n_ch;
        let t = i / n_ch;
        let v = *channels.get(c).and_then(|ch| ch.get(t))?;
        write_int_le(v, elem, &mut out);
    }
    (out.len() == orig_nbytes).then_some(out)
}

/// `can_apply`: accept only little-endian integer datasets; skip everything else
/// so non-integer data is never routed through (and corrupted by) LML.
extern "C" fn can_apply(_dcpl: hid_t, type_id: hid_t, _space: hid_t) -> c_int {
    unsafe {
        let is_int = H5Tget_class(type_id) == H5T_INTEGER;
        let is_le = H5Tget_order(type_id) == H5T_ORDER_LE;
        let sz = H5Tget_size(type_id);
        if is_int && is_le && matches!(sz, 1 | 2 | 4 | 8) {
            1
        } else {
            0
        }
    }
}

/// `set_local`: stash element size, signedness, and channel count (fastest dim
/// of the chunk) into the filter's `cd_values` so `forward` can de-interleave.
extern "C" fn set_local(dcpl: hid_t, type_id: hid_t, _space: hid_t) -> c_int {
    unsafe {
        let elem = H5Tget_size(type_id);
        if elem == 0 {
            return -1;
        }
        let signed = (H5Tget_sign(type_id) == H5T_SGN_2) as c_uint;

        // Channel count = fastest-varying chunk dimension; 1 for rank-1 chunks.
        let mut dims = [0u64; 32];
        let ndims = H5Pget_chunk(dcpl, dims.len() as c_int, dims.as_mut_ptr());
        let n_ch: c_uint = if ndims >= 2 {
            dims[ndims as usize - 1] as c_uint
        } else {
            1
        };

        let cd: [c_uint; 3] = [elem as c_uint, signed, n_ch];
        // H5Z_FLAG_MANDATORY (0): we can losslessly handle every integer chunk
        // can_apply admitted, so there is never a reason to make it optional.
        H5Pmodify_filter(dcpl, LML_H5_FILTER_ID, 0, cd.len(), cd.as_ptr())
    }
}

/// The H5Z filter callback. Panics are caught and turned into the `0` error
/// sentinel — unwinding across the C boundary into libhdf5 would be UB.
unsafe extern "C" fn filter(
    flags: c_uint,
    cd_nelmts: usize,
    cd_values: *const c_uint,
    nbytes: usize,
    buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    let result = catch_unwind(|| {
        if buf.is_null() || (*buf).is_null() || nbytes == 0 {
            return 0usize;
        }
        let input = std::slice::from_raw_parts(*buf as *const u8, nbytes);

        let out: Vec<u8> = if flags & H5Z_FLAG_REVERSE != 0 {
            match reverse(input) {
                Some(v) => v,
                None => return 0,
            }
        } else {
            // cd_values = [elem, signed, n_ch]; fall back to the chunk header's
            // own copy is unnecessary on encode (set_local always ran first).
            if cd_nelmts < 3 {
                return 0;
            }
            let cd = std::slice::from_raw_parts(cd_values, cd_nelmts);
            let (elem, signed, n_ch) = (cd[0] as usize, cd[1] != 0, cd[2] as usize);
            match forward(input, elem, signed, n_ch) {
                Some(v) => v,
                None => return 0,
            }
        };

        // Hand HDF5 a fresh buffer; free the one it gave us.
        let new = H5allocate_memory(out.len(), 0);
        if new.is_null() {
            return 0;
        }
        std::ptr::copy_nonoverlapping(out.as_ptr(), new as *mut u8, out.len());
        H5free_memory(*buf);
        *buf = new;
        *buf_size = out.len();
        out.len()
    });
    result.unwrap_or(0)
}

/// `H5Z_class2_t` holds a `*const c_char` (the name), so it is not `Sync`. The
/// struct is immutable, read-only data handed to libhdf5; wrapping it lets us
/// keep it in a `static`. Safe: never mutated, the `name` points at a `'static`
/// byte string.
struct FilterClass(H5Z_class2_t);
unsafe impl Sync for FilterClass {}

static LML_FILTER_CLASS: FilterClass = FilterClass(H5Z_class2_t {
    version: H5Z_CLASS_T_VERS as c_int,
    id: LML_H5_FILTER_ID,
    encoder_present: 1,
    decoder_present: 1,
    name: FILTER_NAME.as_ptr() as *const std::os::raw::c_char,
    can_apply: Some(can_apply),
    set_local: Some(set_local),
    filter: Some(filter),
});

/// Register the LML filter with libhdf5 **in-process** — the alternative to
/// dynamic plugin discovery, for embedders and for tests. Returns the HDF5
/// `herr_t` (>= 0 on success). After this, datasets created with filter id
/// [`LML_H5_FILTER_ID`] compress through LML without `HDF5_PLUGIN_PATH`.
pub fn register_lml_filter() -> c_int {
    unsafe { H5Zregister(&LML_FILTER_CLASS.0 as *const H5Z_class2_t as *const c_void) }
}

// ── HDF5 plugin discovery entry points (exported C symbols) ─────────────────

/// HDF5 calls this to learn the plugin kind. Returns `H5PL_TYPE_FILTER` (the
/// `H5PL_type_t` enum is ABI-identical to a C `int`).
#[no_mangle]
pub extern "C" fn H5PLget_plugin_type() -> c_int {
    H5PL_TYPE_FILTER
}

/// HDF5 calls this to fetch the filter class struct.
#[no_mangle]
pub extern "C" fn H5PLget_plugin_info() -> *const c_void {
    &LML_FILTER_CLASS.0 as *const H5Z_class2_t as *const c_void
}

#[cfg(test)]
mod abi_check {
    /// Our hand-declared `H5Z_class2_t` (link-free, see `hdf5_abi`) must have the
    /// exact size + alignment of the real one, or libhdf5 would read garbage
    /// through the filter-class pointer. Cross-check against `hdf5-metno-sys`
    /// (dev-dependency only — never linked into the shipped cdylib).
    #[test]
    fn h5z_class2_t_layout_matches_sys() {
        assert_eq!(
            std::mem::size_of::<super::hdf5_abi::H5Z_class2_t>(),
            std::mem::size_of::<hdf5_metno_sys::h5z::H5Z_class2_t>(),
            "H5Z_class2_t size drift vs libhdf5 bindings"
        );
        assert_eq!(
            std::mem::align_of::<super::hdf5_abi::H5Z_class2_t>(),
            std::mem::align_of::<hdf5_metno_sys::h5z::H5Z_class2_t>(),
            "H5Z_class2_t alignment drift vs libhdf5 bindings"
        );
    }
}
