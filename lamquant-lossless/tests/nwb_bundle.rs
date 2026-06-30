//! Zero-skeleton NWB ⇄ SignalBundle round-trip (ADR 0051 Track 3, Phase B).
//!
//! Proves the headline claim: integer datasets round-trip byte-exact through the
//! bundle, AND everything LML doesn't touch — float datasets, attributes, and
//! **object references** (the hard case a structural transcoder breaks) —
//! survives, because the skeleton is a real HDF5 file with only the integer
//! payloads zeroed. h5py authors + verifies (real ecosystem tooling). Skips when
//! python3+h5py absent.
#![cfg(feature = "nwb")]

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

    let bundle = lamquant_core::nwb::read_bundle(&src).expect("read_bundle");
    // sidecar carries skeleton + slots; signal carries the two integer datasets.
    assert!(bundle.sidecar.iter().any(|s| s.key == "nwb_skeleton"));
    assert!(!bundle.signal.is_empty());
    lamquant_core::nwb::write_bundle(&bundle, &out).expect("write_bundle");

    match py(CHECK, &[&src, &out]) {
        Some(true) => {}
        Some(false) => panic!("round-trip changed data or structure"),
        None => {}
    }
}
