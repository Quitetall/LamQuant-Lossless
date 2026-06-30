//! End-to-end round-trip of the LML H5Z filter through **real libhdf5**.
//!
//! Registers the filter in-process (`H5Zregister`) — same system libhdf5 the
//! filter links, so there is no dual-HDF5 ABI mismatch — then writes a chunked
//! integer dataset through the filter, reads it back, and asserts:
//!   * the data is byte-identical after the LML encode→decode chunk pipeline;
//!   * the filter is recorded in the dataset's pipeline;
//!   * the filtered file is smaller than the same data stored uncompressed
//!     (the filter actually compressed, not just passed through).

use hdf5_metno::File;
use lamquant_lml_h5filter::{register_lml_filter, LML_H5_FILTER_ID};

fn write_dataset(path: &std::path::Path, data: &ndarray::Array2<i16>, filtered: bool) {
    let f = File::create(path).unwrap();
    let mut b = f.new_dataset_builder().chunk((250usize, data.shape()[1]));
    if filtered {
        b = b.add_filter(LML_H5_FILTER_ID, &[]); // cd_values filled by set_local
    }
    b.with_data(data).create("signal").unwrap();
}

#[test]
fn filter_roundtrips_int16_dataset_through_libhdf5() {
    assert!(register_lml_filter() >= 0, "H5Zregister failed");

    let dir = tempfile::tempdir().unwrap();
    let filt = dir.path().join("filtered.h5");
    let raw = dir.path().join("raw.h5");

    // 1000 x 4 int16, per-channel smooth ramps → compressible biosignal-like data.
    let (t, c) = (1000usize, 4usize);
    let data = ndarray::Array2::from_shape_fn((t, c), |(i, j)| {
        (((i as f64) * 0.5).sin() * 1000.0) as i16 + (j as i16) * 7
    });

    write_dataset(&filt, &data, true);
    write_dataset(&raw, &data, false);

    // Reopen the filtered file and read back through the reverse (decode) path.
    let f = File::open(&filt).unwrap();
    let ds = f.dataset("signal").unwrap();
    let back = ds.read_2d::<i16>().unwrap();
    assert_eq!(back, data, "round-trip mismatch through the LML H5Z filter");

    // The LML filter must be in the dataset's pipeline.
    assert!(
        ds.filters().iter().any(|fl| fl.id() == LML_H5_FILTER_ID),
        "LML filter not recorded in the dataset pipeline: {:?}",
        ds.filters()
    );
    drop(f);

    // The filter actually compressed: filtered file < unfiltered file.
    let fs_filt = std::fs::metadata(&filt).unwrap().len();
    let fs_raw = std::fs::metadata(&raw).unwrap().len();
    assert!(
        fs_filt < fs_raw,
        "filter did not shrink the file: filtered {fs_filt} vs raw {fs_raw}"
    );
}
