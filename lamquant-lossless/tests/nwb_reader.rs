//! Cross-tool NWB/HDF5 reader test (ADR 0051 Track 3, Phase 1).
//!
//! The fixture is authored by **h5py** — the actual ecosystem tool — so this
//! proves real NWB/HDF5 interop, not just round-trip through our own writer.
//! It builds an NWB-shaped file (`/acquisition/ElectricalSeries/data`, int16,
//! time-major `(T, C)`), plus a uint8 1-D dataset and a float64 dataset, then
//! checks `nwb::read_int_signals`:
//!   * int datasets are extracted, widened to i64, transposed to channel-major;
//!   * the float dataset is skipped (LML is integer-only);
//!   * on-disk width / signedness / orientation / shape are reported faithfully.
//!
//! Skips (does not fail) when python3 + h5py are unavailable, so CI without the
//! Python toolchain stays green while the check runs wherever h5py exists.
#![cfg(feature = "nwb")]

use std::path::Path;
use std::process::Command;

/// Author the fixture with h5py. Returns false if python3/h5py is unavailable.
fn write_fixture_with_h5py(path: &Path) -> bool {
    // data[t][c] = t*10 + c  (int16, shape (1000, 4), time-major NWB layout)
    // pulse (uint8, shape (256,)) : pulse[i] = i % 7
    // volts (float64, shape (1000, 4)) : must be SKIPPED by the reader
    let script = format!(
        r#"
import sys
try:
    import h5py, numpy as np
except Exception:
    sys.exit(42)
T, C = 1000, 4
data = (np.arange(T).reshape(T,1)*10 + np.arange(C).reshape(1,C)).astype('<i2')
pulse = (np.arange(256) % 7).astype('u1')
volts = data.astype('<f8') * 0.5
with h5py.File(r"{}", "w") as f:
    es = f.create_group("acquisition").create_group("ElectricalSeries")
    es.create_dataset("data", data=data)
    f.create_dataset("pulse", data=pulse)
    es.create_dataset("volts", data=volts)
sys.exit(0)
"#,
        path.display()
    );
    match Command::new("python3").arg("-c").arg(&script).status() {
        Ok(s) if s.success() => true,
        Ok(s) if s.code() == Some(42) => {
            eprintln!("SKIP nwb_reader: h5py not available");
            false
        }
        Ok(s) => panic!("h5py fixture generation failed: {s:?}"),
        Err(e) => {
            eprintln!("SKIP nwb_reader: python3 not runnable: {e}");
            false
        }
    }
}

#[test]
fn reads_h5py_authored_nwb_int_datasets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let h5 = dir.path().join("fixture.nwb");
    if !write_fixture_with_h5py(&h5) {
        return; // toolchain absent — skip cleanly
    }

    let sigs = lamquant_core::nwb::read_int_signals(&h5).expect("read_int_signals");

    // Two integer datasets (data, pulse); the float64 `volts` must be skipped.
    assert_eq!(
        sigs.len(),
        2,
        "expected exactly the two integer datasets, got {:?}",
        sigs.iter().map(|s| &s.h5_path).collect::<Vec<_>>()
    );

    let data = sigs
        .iter()
        .find(|s| s.h5_path.ends_with("ElectricalSeries/data"))
        .expect("ElectricalSeries/data present");
    assert_eq!(data.int_bytes, 2);
    assert!(data.signed);
    assert!(
        data.time_major,
        "NWB (T,C) layout must be reported time-major"
    );
    assert_eq!(data.orig_shape, vec![1000, 4]);
    // channel-major: 4 channels, each 1000 samples; sig[c][t] = t*10 + c.
    assert_eq!(data.signal.len(), 4);
    assert_eq!(data.signal[0].len(), 1000);
    for c in 0..4 {
        for t in 0..1000 {
            assert_eq!(
                data.signal[c][t],
                (t as i64) * 10 + c as i64,
                "value mismatch at channel {c}, sample {t}"
            );
        }
    }

    let pulse = sigs
        .iter()
        .find(|s| s.h5_path.ends_with("/pulse"))
        .expect("pulse present");
    assert_eq!(pulse.int_bytes, 1);
    assert!(!pulse.signed);
    assert!(
        !pulse.time_major,
        "1-D dataset is a single channel, no transpose"
    );
    assert_eq!(pulse.orig_shape, vec![256]);
    assert_eq!(pulse.signal.len(), 1);
    for i in 0..256 {
        assert_eq!(pulse.signal[0][i], (i % 7) as i64);
    }
}
