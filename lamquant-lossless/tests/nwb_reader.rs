//! Cross-tool NWB/HDF5 reader test (ADR 0051 Track 3, Phase 1).
//!
//! The fixture is authored by **h5py** — the actual ecosystem tool — so this
//! proves real NWB/HDF5 interop, not just round-trip through our own writer.
//! It builds an NWB-shaped file (`/acquisition/ElectricalSeries/data`, int16,
//! time-major `(T, C)`), plus a uint8 1-D dataset and a float64 dataset, then
//! checks `nwb::read_semantic`:
//!   * int datasets are extracted, widened to i64, transposed to channel-major;
//!   * the float dataset is skipped (LML is integer-only);
//!   * on-disk width / signedness / orientation / shape are reported faithfully.
//!
//! Skips (does not fail) when python3 + h5py are unavailable, so CI without the
//! Python toolchain stays green while the check runs wherever h5py exists.
#![cfg(feature = "nwb")]

use serde::Deserialize;
use std::collections::HashMap;
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

#[derive(Debug, Deserialize)]
struct NwbSlot {
    h5_path: String,
    int_bytes: u8,
    signed: bool,
    orig_shape: Vec<usize>,
    time_major: bool,
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
        let bytes = block.bytes();
        assert_eq!(bytes.len() % 8, 0, "NWB tensor payload is i64 bytes");
        signal.push(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("exact chunk")))
                .collect(),
        );
    }
    signal
}

#[test]
fn reads_h5py_authored_nwb_int_datasets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let h5 = dir.path().join("fixture.nwb");
    if !write_fixture_with_h5py(&h5) {
        return; // toolchain absent — skip cleanly
    }

    let read = lamquant_core::nwb::read_semantic(&h5).expect("read_semantic");
    let dataset = read.opened.dataset();
    assert_eq!(dataset.recordings().len(), 1);
    assert_eq!(dataset.streams().len(), 1);
    assert_eq!(dataset.source_capsules().len(), 2);
    assert_eq!(
        dataset.source_capsules()[0].source().namespace(),
        "source.nwb.capsule.0"
    );
    assert_eq!(
        dataset.source_capsules()[1].source().namespace(),
        "source.nwb.capsule.1"
    );
    let signal = signal_from_read(&read);
    let slots = nwb_slots(&read);
    let mut by_path = HashMap::with_capacity(slots.len());
    for slot in slots {
        by_path.insert(slot.h5_path.clone(), slot);
    }

    // Two integer datasets (data, pulse); the float64 `volts` must be skipped.
    assert_eq!(
        by_path.len(),
        2,
        "expected exactly the two integer datasets, got {:?}",
        by_path.keys().collect::<Vec<_>>()
    );

    let data = by_path
        .remove("/acquisition/ElectricalSeries/data")
        .expect("ElectricalSeries/data present");
    assert_eq!(data.int_bytes, 2, "ElectricalSeries/data width");
    assert!(data.signed);
    assert!(
        data.time_major,
        "NWB (T,C) layout must be reported time-major"
    );
    assert_eq!(
        data.orig_shape,
        vec![1000, 4],
        "ElectricalSeries/data shape"
    );
    assert_eq!(data.n_ch, 4);
    let data_signal = &signal[data.first_ch..data.first_ch + data.n_ch];
    // channel-major: 4 channels, each 1000 samples; sig[c][t] = t*10 + c.
    assert_eq!(data_signal.len(), 4);
    assert_eq!(data_signal[0].len(), 1000);
    for (c, channel) in data_signal.iter().enumerate() {
        for (t, value) in channel.iter().copied().enumerate() {
            assert_eq!(
                value,
                (t as i64) * 10 + c as i64,
                "value mismatch at channel {c}, sample {t}"
            );
        }
    }

    let pulse = by_path.remove("/pulse").expect("pulse present");
    assert_eq!(pulse.int_bytes, 1);
    assert!(!pulse.signed);
    assert!(!pulse.time_major, "1-D dataset has no transpose");
    assert_eq!(pulse.orig_shape, vec![256], "pulse shape");
    assert_eq!(pulse.n_ch, 1);
    let pulse_signal = &signal[pulse.first_ch..pulse.first_ch + pulse.n_ch];
    assert_eq!(pulse_signal.len(), 1);
    for (i, value) in pulse_signal[0].iter().copied().enumerate() {
        assert_eq!(value, (i % 7) as i64);
    }
}
