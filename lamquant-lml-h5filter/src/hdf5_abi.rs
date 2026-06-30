//! Minimal, **link-free** HDF5 ABI surface used by the filter.
//!
//! Why hand-declared instead of `hdf5-metno-sys`: an HDF5 filter plugin must be
//! loadable by *whatever* libhdf5 loaded it (the system one, conda's, or the
//! libhdf5 bundled inside a pip `h5py` wheel). If the `.so` carried a hard
//! `DT_NEEDED` on a specific libhdf5, loading it into a process that already has
//! a *different* libhdf5 would pull a second HDF5 into the address space —
//! mismatched `hid_t` handles, allocator splits, corruption. So we declare the
//! ~10 functions/types we touch in plain `extern "C"` blocks with **no `#[link]`
//! attribute**: the symbols stay undefined in the cdylib and resolve at
//! `dlopen` time against the host's libhdf5. (A `.so` with undefined symbols is
//! normal and valid on ELF platforms.)
//!
//! Layout/signatures mirror HDF5's public headers exactly; `tests/abi.rs`
//! cross-checks `H5Z_class2_t`'s size/alignment against `hdf5-metno-sys`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type hid_t = i64;
pub type herr_t = c_int;
pub type htri_t = c_int;
pub type hsize_t = u64;
pub type H5Z_filter_t = c_int;

// Constants (from H5Zpublic.h / H5Tpublic.h / H5PLpublic.h).
pub const H5Z_CLASS_T_VERS: c_int = 1;
pub const H5Z_FLAG_REVERSE: c_uint = 0x0100;
pub const H5T_INTEGER: c_int = 0; // H5T_class_t::H5T_INTEGER
pub const H5T_ORDER_LE: c_int = 0; // H5T_order_t::H5T_ORDER_LE
pub const H5T_SGN_2: c_int = 1; // H5T_sign_t::H5T_SGN_2 (signed two's complement)
pub const H5PL_TYPE_FILTER: c_int = 0; // H5PL_type_t::H5PL_TYPE_FILTER

/// `H5Z_class2_t` — the filter class descriptor (must match HDF5's layout).
#[repr(C)]
pub struct H5Z_class2_t {
    pub version: c_int,
    pub id: H5Z_filter_t,
    pub encoder_present: c_uint,
    pub decoder_present: c_uint,
    pub name: *const c_char,
    pub can_apply: Option<extern "C" fn(hid_t, hid_t, hid_t) -> htri_t>,
    pub set_local: Option<extern "C" fn(hid_t, hid_t, hid_t) -> herr_t>,
    pub filter: Option<
        unsafe extern "C" fn(
            flags: c_uint,
            cd_nelmts: usize,
            cd_values: *const c_uint,
            nbytes: usize,
            buf_size: *mut usize,
            buf: *mut *mut c_void,
        ) -> usize,
    >,
}

// NO `#[link(...)]` here — symbols resolve from the loading process's libhdf5.
extern "C" {
    pub fn H5allocate_memory(size: usize, clear: c_uint) -> *mut c_void;
    pub fn H5free_memory(mem: *mut c_void) -> herr_t;
    pub fn H5Pget_chunk(plist_id: hid_t, max_ndims: c_int, dim: *mut hsize_t) -> c_int;
    pub fn H5Pmodify_filter(
        plist_id: hid_t,
        filter: H5Z_filter_t,
        flags: c_uint,
        cd_nelmts: usize,
        cd_values: *const c_uint,
    ) -> herr_t;
    pub fn H5Tget_class(type_id: hid_t) -> c_int; // H5T_class_t
    pub fn H5Tget_order(type_id: hid_t) -> c_int; // H5T_order_t
    pub fn H5Tget_sign(type_id: hid_t) -> c_int; // H5T_sign_t
    pub fn H5Tget_size(type_id: hid_t) -> usize;
    pub fn H5Zregister(cls: *const c_void) -> herr_t;
}
