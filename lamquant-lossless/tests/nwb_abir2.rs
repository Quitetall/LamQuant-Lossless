#![cfg(feature = "nwb")]

use abir::{decode_bcs2, encode_bcs2, SampleBuffer, TensorDataType};
use hdf5_metno::File;

fn write_mixed_rate_nwb(path: &std::path::Path) {
    let file = File::create(path).unwrap();
    let acquisition = file.create_group("acquisition").unwrap();

    let fast = acquisition.create_group("FastElectricalSeries").unwrap();
    let fast_values =
        ndarray::Array2::from_shape_vec((4, 2), vec![1_i16, 10, 2, 20, 3, 30, 4, 40]).unwrap();
    fast.new_dataset_builder()
        .with_data(&fast_values)
        .create("data")
        .unwrap();
    let starting_time = fast.new_dataset::<f64>().create("starting_time").unwrap();
    starting_time.write_scalar(&0.5).unwrap();
    starting_time
        .new_attr::<f64>()
        .create("rate")
        .unwrap()
        .write_scalar(&250.0)
        .unwrap();

    let slow = acquisition.create_group("SlowElectricalSeries").unwrap();
    let slow_values = ndarray::Array2::from_shape_vec((3, 1), vec![100_i32, 200, 300]).unwrap();
    slow.new_dataset_builder()
        .with_data(&slow_values)
        .create("data")
        .unwrap();
    slow.new_dataset_builder()
        .with_data(&[1.0_f64, 1.5, 2.25])
        .create("timestamps")
        .unwrap();

    let float_series = acquisition.create_group("FloatElectricalSeries").unwrap();
    let float_values =
        ndarray::Array2::from_shape_vec((3, 1), vec![0.25_f32, -1.5, f32::from_bits(0x7fc0_1234)])
            .unwrap();
    float_series
        .new_dataset_builder()
        .with_data(&float_values)
        .create("data")
        .unwrap();
    let float_start = float_series
        .new_dataset::<f64>()
        .create("starting_time")
        .unwrap();
    float_start.write_scalar(&3.0).unwrap();
    float_start
        .new_attr::<f64>()
        .create("rate")
        .unwrap()
        .write_scalar(&10.0)
        .unwrap();

    let general = file.create_group("general").unwrap();
    let trials = general.create_group("trials").unwrap();
    trials
        .new_dataset_builder()
        .with_data(&[7_u16, 8])
        .create("id")
        .unwrap();
}

#[test]
fn nwb_lowers_mixed_rate_series_and_auxiliary_integer_tensors() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mixed.nwb");
    write_mixed_rate_nwb(&path);

    let recording = lamquant_core::nwb::read_recording(&path).unwrap();

    assert_eq!(recording.clocks().len(), 1);
    assert_eq!(recording.clocks()[0].tick_rate().numerator(), 1_000_000_000);
    assert_eq!(recording.signal_streams().len(), 1);
    let series = recording.signal_streams()[0].series();
    assert_eq!(series.len(), 4);

    let fast = series
        .iter()
        .find(|series| series.channel().label().contains("FastElectricalSeries[0]"))
        .unwrap();
    assert_eq!(fast.time_axis().start_tick(), Some(500_000_000));
    assert_eq!(fast.time_axis().sample_rate().unwrap().numerator(), 250);
    assert!(matches!(
        fast.samples(),
        SampleBuffer::I16(values) if values.as_ref() == [1, 2, 3, 4]
    ));

    let slow = series
        .iter()
        .find(|series| series.channel().label().contains("SlowElectricalSeries[0]"))
        .unwrap();
    assert_eq!(
        slow.time_axis().explicit_ticks().unwrap(),
        &[1_000_000_000, 1_500_000_000, 2_250_000_000]
    );
    assert!(matches!(
        slow.samples(),
        SampleBuffer::I32(values) if values.as_ref() == [100, 200, 300]
    ));

    let float_series = series
        .iter()
        .find(|series| {
            series
                .channel()
                .label()
                .contains("FloatElectricalSeries[0]")
        })
        .unwrap();
    assert_eq!(float_series.time_axis().start_tick(), Some(3_000_000_000));
    assert_eq!(
        float_series.time_axis().sample_rate().unwrap().numerator(),
        10
    );
    let SampleBuffer::F32(float_values) = float_series.samples() else {
        panic!("float NWB series must retain f32 storage");
    };
    assert_eq!(
        float_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0.25_f32.to_bits(), (-1.5_f32).to_bits(), 0x7fc0_1234]
    );

    assert_eq!(recording.tensors().len(), 1);
    assert_eq!(recording.tensors()[0].shape(), &[2]);
    assert_eq!(recording.tensors()[0].buffer().dtype(), TensorDataType::U16);
    assert_eq!(recording.attachments().len(), 2);
    let skeleton = recording
        .attachments()
        .iter()
        .find(|attachment| attachment.id() == "attachment:nwb:skeleton-zstd")
        .unwrap();
    let decoded = zstd::stream::decode_all(skeleton.bytes()).unwrap();
    assert_eq!(&decoded[..8], b"\x89HDF\r\n\x1a\n");
    assert!(recording.verify().is_ok());
}

#[test]
fn nwb_recording_is_deterministic_through_bcs2() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mixed.nwb");
    write_mixed_rate_nwb(&path);

    let first = lamquant_core::nwb::read_recording(&path).unwrap();
    let second = lamquant_core::nwb::read_recording(&path).unwrap();
    let bytes = encode_bcs2(&first).unwrap();
    assert_eq!(bytes, encode_bcs2(&second).unwrap());
    let decoded = decode_bcs2(&bytes).unwrap();
    assert_eq!(decoded.signal_streams()[0].series().len(), 4);
    assert_eq!(decoded.tensors().len(), 1);

    let restored_path = directory.path().join("restored.nwb");
    lamquant_core::nwb::write_recording(&decoded, &restored_path).unwrap();
    let published_bytes = std::fs::read(&restored_path).unwrap();
    let error = lamquant_core::nwb::write_recording(&decoded, &restored_path).unwrap_err();
    assert!(error.to_string().contains("already exists"));
    assert_eq!(std::fs::read(&restored_path).unwrap(), published_bytes);
    let original_datasets = lamquant_core::nwb::read_int_signals(&path).unwrap();
    let restored_datasets = lamquant_core::nwb::read_int_signals(&restored_path).unwrap();
    assert_eq!(original_datasets.len(), restored_datasets.len());
    for (original, restored) in original_datasets.iter().zip(&restored_datasets) {
        assert_eq!(original.h5_path, restored.h5_path);
        assert_eq!(original.signal, restored.signal);
        assert_eq!(original.int_bytes, restored.int_bytes);
        assert_eq!(original.signed, restored.signed);
        assert_eq!(original.orig_shape, restored.orig_shape);
    }
    let restored_file = File::open(&restored_path).unwrap();
    let restored_float = restored_file
        .dataset("/acquisition/FloatElectricalSeries/data")
        .unwrap()
        .read_2d::<f32>()
        .unwrap();
    assert_eq!(
        restored_float
            .as_slice()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0.25_f32.to_bits(), (-1.5_f32).to_bits(), 0x7fc0_1234]
    );
    let restored_recording = lamquant_core::nwb::read_recording(&restored_path).unwrap();
    let restored_slow = restored_recording.signal_streams()[0]
        .series()
        .iter()
        .find(|series| series.channel().label().contains("SlowElectricalSeries[0]"))
        .unwrap();
    assert_eq!(
        restored_slow.time_axis().explicit_ticks().unwrap(),
        &[1_000_000_000, 1_500_000_000, 2_250_000_000]
    );
}

#[test]
fn nwb_timeseries_without_timing_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing-timing.nwb");
    {
        let file = File::create(&path).unwrap();
        let acquisition = file.create_group("acquisition").unwrap();
        let series = acquisition.create_group("ElectricalSeries").unwrap();
        series
            .new_dataset_builder()
            .with_data(&ndarray::Array2::from_shape_vec((2, 1), vec![1_i16, 2]).unwrap())
            .create("data")
            .unwrap();
    }

    let error = lamquant_core::nwb::read_recording(&path).unwrap_err();
    assert!(error
        .to_string()
        .contains("missing timestamps or starting_time/rate"));
}

#[test]
fn nwb_timestamp_count_mismatch_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad-timestamps.nwb");
    {
        let file = File::create(&path).unwrap();
        let acquisition = file.create_group("acquisition").unwrap();
        let series = acquisition.create_group("ElectricalSeries").unwrap();
        series
            .new_dataset_builder()
            .with_data(&ndarray::Array2::from_shape_vec((3, 1), vec![1_i16, 2, 3]).unwrap())
            .create("data")
            .unwrap();
        series
            .new_dataset_builder()
            .with_data(&[0.0_f64, 0.5])
            .create("timestamps")
            .unwrap();
    }

    let error = lamquant_core::nwb::read_recording(&path).unwrap_err();
    assert!(error.to_string().contains("timestamp count"));
}
