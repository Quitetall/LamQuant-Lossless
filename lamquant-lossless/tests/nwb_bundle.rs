//! Zero-skeleton NWB ⇄ ABIR Tensor round-trip (ADR 0051 Track 3, Phase B).
//!
//! Proves the headline claim: integer datasets round-trip byte-exact through the
//! bundle, AND everything LML doesn't touch — float datasets, attributes, and
//! **object references** (the hard case a structural transcoder breaks) —
//! survives, because the skeleton is a real HDF5 file with only the integer
//! payloads zeroed. h5py authors + verifies (real ecosystem tooling). Skips when
//! python3+h5py absent.
#![cfg(feature = "nwb")]

use serde::Deserialize;
use std::path::Path;
use std::process::Command;

fn py(script: &str, args: &[&Path]) -> Option<bool> {
    let mut c = Command::new("python3");
    c.arg("-c").arg(script);
    for a in args {
        c.arg(a);
    }
    match c.status() {
        Ok(s) if s.code() == Some(42) => {
            eprintln!("SKIP nwb_bundle: h5py unavailable");
            None
        }
        Ok(s) => Some(s.success()),
        Err(e) => {
            eprintln!("SKIP nwb_bundle: python3 not runnable: {e}");
            None
        }
    }
}

const MAKE: &str = r#"
import sys
try:
    import h5py, numpy as np
except Exception:
    sys.exit(42)
T, C = 500, 4
data = (np.arange(T).reshape(T,1)*10 + np.arange(C).reshape(1,C)).astype('<i2')
with h5py.File(sys.argv[1], "w") as f:
    f.attrs["nwb_version"] = "2.6.0"
    es = f.create_group("acquisition").create_group("ElectricalSeries")
    d = es.create_dataset("data", data=data, chunks=(125, C))
    es.attrs["unit"] = "volts"
    f.create_group("general").create_dataset("volts", data=data.astype('<f8')*0.5)
    es.attrs["data_ref"] = d.ref            # object reference — the hard case
    f.create_dataset("flags", data=(np.arange(256) % 5).astype('u1'))
sys.exit(0)
"#;

const CHECK: &str = r#"
import sys, numpy as np, h5py
a, b = h5py.File(sys.argv[1],"r"), h5py.File(sys.argv[2],"r")
ok = True
def eq(name, x, y):
    global ok
    good = np.array_equal(x, y)
    ok = ok and good
    if not good: print("MISMATCH:", name)
eq("data",  a["acquisition/ElectricalSeries/data"][...], b["acquisition/ElectricalSeries/data"][...])
eq("flags", a["flags"][...], b["flags"][...])
eq("volts(float)", a["general/volts"][...], b["general/volts"][...])
if b.attrs.get("nwb_version") != "2.6.0": print("attr nwb_version lost"); ok=False
if b["acquisition/ElectricalSeries"].attrs.get("unit") != "volts": print("attr unit lost"); ok=False
# object reference must still resolve to the (refilled) data
ref = b["acquisition/ElectricalSeries"].attrs["data_ref"]
deref = b[ref][...]
eq("deref(object ref)", deref, b["acquisition/ElectricalSeries/data"][...])
sys.exit(0 if ok else 1)
"#;

#[derive(Debug, Deserialize)]
struct NwbSlot {
    h5_path: String,
    first_ch: usize,
    n_ch: usize,
}

fn nwb_slots(read: &lamquant_core::source::SemanticRead) -> Vec<NwbSlot> {
    let capsule = read
        .opened
        .dataset()
        .source_capsules()
        .iter()
        .find(|capsule| {
            capsule.source().namespace() == "source.nwb.capsule.1"
                && capsule.source().value() == "nwb_slots"
        })
        .expect("slot metadata capsule present");
    let bytes = read
        .opened
        .access()
        .payload_bytes(capsule.content_id())
        .expect("slot metadata payload available");
    serde_json::from_slice(bytes).expect("slot metadata decodes")
}

fn signal_from_read(read: &lamquant_core::source::SemanticRead) -> Vec<Vec<i64>> {
    let mut signal = Vec::with_capacity(read.mapping.channels.len());
    for channel in &read.mapping.channels {
        let block = read
            .opened
            .block_view(channel.atom_id)
            .expect("NWB tensor payload block view");
        assert_eq!(
            block.bytes().len() % 8,
            0,
            "NWB tensor payload is i64 bytes"
        );
        signal.push(
            block
                .bytes()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("exact chunk")))
                .collect(),
        );
    }
    signal
}

#[test]
fn zero_skeleton_roundtrip_preserves_structure_and_data() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("in.nwb");
    let out = dir.path().join("out.nwb");

    match py(MAKE, &[&src]) {
        Some(true) => {}
        Some(false) => panic!("h5py fixture generation failed"),
        None => return, // toolchain absent — skip
    }

    let read = lamquant_core::nwb::read_semantic(&src).expect("read_semantic");
    let dataset = read.opened.dataset();
    assert_eq!(read.mapping.source_capsule_count, 2);
    assert_eq!(dataset.source_capsules().len(), 2);
    assert_eq!(dataset.recordings().len(), 1);
    assert_eq!(dataset.streams().len(), 1);

    let slots = nwb_slots(&read);
    assert!(!slots.is_empty());
    for slot in &slots {
        assert!(!slot.h5_path.is_empty());
        assert!(slot.first_ch < read.mapping.channel_count);
        assert!(slot.n_ch > 0);
        assert!(slot.first_ch + slot.n_ch <= read.mapping.channel_count);
    }

    let signal = signal_from_read(&read);
    let stream_atoms = &dataset.streams()[0].atoms();
    for (channel, samples) in read.mapping.channels.iter().zip(signal.iter()) {
        assert!(!samples.is_empty());
        assert!(stream_atoms.contains(&channel.atom_id));
    }
    let data_slot = slots
        .iter()
        .find(|slot| slot.h5_path.ends_with("ElectricalSeries/data"))
        .expect("data slot present");
    assert_eq!(data_slot.n_ch, 4, "data slot should be 4 channels");

    lamquant_core::nwb::write_semantic(&read, &out).expect("write_semantic");

    match py(CHECK, &[&src, &out]) {
        Some(true) => {}
        Some(false) => panic!("round-trip changed data or structure"),
        None => {}
    }
}
