use serde_json::Value;
use sha2::{Digest, Sha256};

use lamquant_lml_optimum_v2::fixed_universal_conformance::{
    FixedRlsExpert, FixedUniversalGraph, UniversalSession,
};

#[test]
fn shared_fixture_graph_matches_compact_wire() {
    let fixture = shared_fixture();
    assert_eq!(
        fixture["schema"].as_str(),
        Some("lamquant.optimum_v2.fixed_universal_conformance.v1")
    );
    assert_eq!(json_i64(&fixture["mode"], "mode"), 0x14);
    assert_eq!(
        fixture["integer_residual_encoding"].as_str(),
        Some("signed little-endian i64, channel-major")
    );

    let graph_case = &fixture["graph"];
    let graph = FixedUniversalGraph::new(json_parents(&graph_case["parents"]))
        .expect("fixture graph is causal");
    let encoded = hex_bytes(
        graph_case["serialized_hex"]
            .as_str()
            .expect("fixture graph serialized_hex is a string"),
    );

    assert_eq!(graph.serialize(), encoded);
    assert_eq!(
        FixedUniversalGraph::parse(&encoded, graph.channel_count()).expect("fixture graph parses"),
        graph
    );
}

#[test]
fn shared_fixture_scalar_state_matches_live_core() {
    let fixture = shared_fixture();
    let scalar = &fixture["scalar_case"];
    assert_eq!(
        json_string_vec(&scalar["expected_row_fields"], "scalar expected_row_fields"),
        [
            "feature",
            "sample",
            "dot_q20",
            "prediction",
            "residual",
            "z_q36",
            "denominator_q36_raw2",
            "gain_q40",
            "error_q20",
            "weight_q20",
            "covariance_q36",
            "score_q20",
        ]
    );
    let mut expert = FixedRlsExpert::new(
        json_usize(&scalar["width"], "scalar width"),
        json_u8(&scalar["bit_depth"], "scalar bit_depth"),
    )
    .expect("fixture scalar expert is bounded");
    let samples = json_i64_vec(&scalar["samples"], "scalar samples");
    let expected_rows = scalar["expected_rows"]
        .as_array()
        .expect("fixture scalar expected_rows is an array");
    assert_eq!(samples.len(), expected_rows.len());

    let mut previous = 0i64;
    for (sample, expected) in samples.into_iter().zip(expected_rows) {
        let features = [previous];
        let dot_q20 = expert.dot_q20(&features).expect("bounded scalar dot");
        let prediction = expert
            .prediction(&features)
            .expect("bounded scalar prediction");
        let trace = expert
            .observe(&features, sample)
            .expect("bounded scalar update");
        let actual = vec![
            i128::from(previous),
            i128::from(sample),
            dot_q20,
            i128::from(prediction),
            i128::from(sample - prediction),
            trace.z_q36[0],
            trace.denominator_q36_raw2,
            trace.gain_q40[0],
            trace.error_q20,
            expert.weights_q20()[0],
            expert.covariance_q36()[0][0],
            expert.score_q20(),
        ];
        assert_eq!(
            actual,
            json_i128_vec(expected, "scalar expected row"),
            "scalar fixture diverged after sample {sample}"
        );
        previous = sample;
    }
}

#[test]
fn shared_fixture_session_residuals_match_live_core() {
    let fixture = shared_fixture();
    let cases = fixture["session_cases"]
        .as_array()
        .expect("fixture session_cases is an array");
    assert_eq!(
        cases.len(),
        2,
        "fixture must cover nominal and i32 boundary cases"
    );
    for case in cases {
        assert_session_case(case);
    }
}

#[test]
fn shared_fixture_covariance_reset_matches_live_core() {
    let fixture = shared_fixture();
    let reset = &fixture["reset_case"];
    let mut expert = FixedRlsExpert::new(
        json_usize(&reset["width"], "reset width"),
        json_u8(&reset["bit_depth"], "reset bit_depth"),
    )
    .expect("fixture reset expert is bounded");
    let features = json_i64_vec(&reset["features"], "reset features");
    let sample = json_i64(&reset["sample"], "reset sample");
    let updates_before_reset =
        json_usize(&reset["updates_before_reset"], "reset updates_before_reset");

    for _ in 0..updates_before_reset {
        let trace = expert
            .observe(&features, sample)
            .expect("bounded pre-reset update");
        assert!(!trace.reset);
    }
    assert_eq!(
        expert.covariance_q36(),
        json_i128_matrix(
            &reset["before_reset"]["covariance_q36"],
            "before covariance"
        )
    );
    assert_eq!(
        expert.reset_count(),
        json_u64(&reset["before_reset"]["reset_count"], "before reset_count")
    );

    let trace = expert
        .observe(&features, sample)
        .expect("bounded reset-triggering update");
    let expected_trace = &reset["reset_trace"];
    assert_eq!(
        trace.z_q36,
        json_i128_vec(&expected_trace["z_q36"], "reset z_q36")
    );
    assert_eq!(
        trace.denominator_q36_raw2,
        json_i128(&expected_trace["denominator_q36_raw2"], "reset denominator")
    );
    assert_eq!(
        trace.gain_q40,
        json_i128_vec(&expected_trace["gain_q40"], "reset gain_q40")
    );
    assert_eq!(
        trace.error_q20,
        json_i128(&expected_trace["error_q20"], "reset error_q20")
    );
    assert_eq!(
        trace.reset,
        expected_trace["reset"]
            .as_bool()
            .expect("reset flag is a boolean")
    );
    assert_eq!(
        expert.weights_q20(),
        json_i128_vec(&reset["after_reset"]["weights_q20"], "after weights")
    );
    assert_eq!(
        expert.covariance_q36(),
        json_i128_matrix(&reset["after_reset"]["covariance_q36"], "after covariance")
    );
    assert_eq!(
        expert.score_q20(),
        json_i128(&reset["after_reset"]["score_q20"], "after score")
    );
    assert_eq!(
        expert.reset_count(),
        json_u64(&reset["after_reset"]["reset_count"], "after reset_count")
    );
}

#[test]
fn fixed_universal_graph_matches_python_compact_wire() {
    let graph = FixedUniversalGraph::new(vec![None, Some(0), Some(1)]).expect("valid graph");

    let encoded = graph.serialize();

    assert_eq!(encoded, hex_bytes("14ffff00000100"));
    assert_eq!(
        FixedUniversalGraph::parse(&encoded, 3).expect("parse compact graph"),
        graph
    );
}

#[test]
fn fixed_rls_scalar_state_matches_python_conformance_vector() {
    let mut expert = FixedRlsExpert::new(1, 16).expect("bounded expert");
    let mut previous = 0i64;
    let expected: [[i128; 12]; 5] = [
        [
            0,
            100,
            0,
            0,
            100,
            0,
            69_544_110_456_832,
            0,
            104_857_600,
            0,
            69_534_332_191,
            104_857_600,
        ],
        [
            100,
            102,
            0,
            0,
            102,
            6_953_433_219_100,
            764_887_432_366_832,
            9_995_432_470,
            106_954_752,
            972_303,
            6_397_076_781,
            210_583_552,
        ],
        [
            102,
            101,
            99_174_906,
            95,
            6,
            652_501_831_662,
            136_099_297_286_356,
            5_271_396_439,
            6_731_270,
            1_004_575,
            3_307_542_864,
            214_847_046,
        ],
        [
            101,
            -50,
            101_462_075,
            97,
            -147,
            334_061_829_264,
            103_284_355_212_496,
            3_556_248_813,
            -153_890_875,
            506_832,
            2_253_464_595,
            366_220_182,
        ],
        [
            -50,
            75,
            -25_341_600,
            -24,
            99,
            -112_673_229_750,
            75_177_771_944_332,
            -1_647_901_009,
            103_984_800,
            350_984,
            2_109_313_292,
            465_913_339,
        ],
    ];

    for (sample, expected_row) in [100i64, 102, 101, -50, 75].into_iter().zip(expected) {
        let features = [previous];
        let dot_q20 = expert.dot_q20(&features).expect("bounded dot product");
        let prediction = expert.prediction(&features).expect("bounded prediction");
        let trace = expert.observe(&features, sample).expect("bounded update");
        assert_eq!(
            [
                i128::from(previous),
                i128::from(sample),
                dot_q20,
                i128::from(prediction),
                i128::from(sample - prediction),
                trace.z_q36[0],
                trace.denominator_q36_raw2,
                trace.gain_q40[0],
                trace.error_q20,
                expert.weights_q20()[0],
                expert.covariance_q36()[0][0],
                expert.score_q20(),
            ],
            expected_row
        );
        previous = sample;
    }
}

#[test]
fn universal_session_matches_cross_language_residual_golden() {
    let samples = [
        (0..96)
            .map(|time| (time * time + 3 * time) % 127 - 63)
            .map(i64::from)
            .collect::<Vec<_>>(),
        (0..96)
            .map(|time| (time * time + 5 * time) % 127 - 61)
            .map(i64::from)
            .collect::<Vec<_>>(),
        (0..96)
            .map(|time| (2 * time * time + time) % 131 - 65)
            .map(i64::from)
            .collect::<Vec<_>>(),
    ];
    let graph = FixedUniversalGraph::new(vec![None, Some(0), Some(1)]).expect("valid graph");
    let mut session = UniversalSession::new(graph, 16).expect("bounded session");
    let mut residuals = vec![Vec::new(); samples.len()];

    for time in 0..96 {
        let mut current = vec![0i64; samples.len()];
        for channel in 0..samples.len() {
            let prediction = session
                .prediction(channel, &current)
                .expect("bounded prediction");
            let sample = samples[channel][time];
            residuals[channel].push(sample - prediction);
            session
                .observe(channel, &current, sample, prediction)
                .expect("bounded observation");
            current[channel] = sample;
        }
        session.finish_time(&current).expect("complete time row");
    }

    let encoded = residuals
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(encoded)),
        "9e2c672b1d8c1c63cdcc222c6ad1610efc0889088b79009e5bd159507e9fabcb"
    );
    assert_eq!(session.reset_count().expect("bounded reset sum"), 0);
}

#[test]
fn fixed_rls_resets_at_python_covariance_limit_golden() {
    let mut expert = FixedRlsExpert::new(1, 16).expect("bounded expert");

    for _ in 0..352 {
        let trace = expert.observe(&[0], 0).expect("bounded silence update");
        assert!(!trace.reset);
    }
    assert_eq!(expert.covariance_q36(), &[vec![4_356_364_193_621]]);
    assert_eq!(expert.reset_count(), 0);

    let trace = expert.observe(&[0], 0).expect("bounded reset update");
    assert_eq!(trace.z_q36, [0]);
    assert_eq!(trace.denominator_q36_raw2, 69_544_110_456_832);
    assert_eq!(trace.gain_q40, [0]);
    assert_eq!(trace.error_q20, 0);
    assert!(trace.reset);
    assert_eq!(expert.weights_q20(), [0]);
    assert_eq!(expert.covariance_q36(), &[vec![68_719_476_736]]);
    assert_eq!(expert.score_q20(), 0);
    assert_eq!(expert.reset_count(), 1);
}

#[test]
fn universal_session_matches_signed_i32_boundary_residual_golden() {
    let samples = (0..96)
        .map(|time| {
            if time % 2 == 0 {
                i64::from(i32::MAX)
            } else {
                i64::from(i32::MIN)
            }
        })
        .collect::<Vec<_>>();
    let graph = FixedUniversalGraph::new(vec![None]).expect("valid graph");
    let mut session = UniversalSession::new(graph, 32).expect("bounded session");
    let mut residuals = Vec::with_capacity(samples.len());

    for sample in samples {
        let mut current = [0i64];
        let prediction = session.prediction(0, &current).expect("prediction");
        residuals.push(sample - prediction);
        session
            .observe(0, &current, sample, prediction)
            .expect("observation");
        current[0] = sample;
        session.finish_time(&current).expect("complete row");
    }

    let encoded = residuals
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(encoded)),
        "25a84096e9048042d1d6c4e50fa2a90a0dfd64e76776a8ec1c53d70eda438772"
    );
    assert_eq!(session.reset_count().expect("bounded reset sum"), 0);
}

#[test]
fn universal_session_enforces_prediction_ticket_ordering() {
    let graph = FixedUniversalGraph::new(vec![None, Some(0)]).expect("valid graph");
    let mut session = UniversalSession::new(graph, 16).expect("bounded session");
    let mut current = [0i64, 0];

    assert!(session.prediction(1, &current).is_err());
    assert!(session.observe(0, &current, 1, 0).is_err());
    let prediction = session.prediction(0, &current).expect("first ticket");
    assert!(session.prediction(0, &current).is_err());
    assert!(session.finish_time(&current).is_err());
    assert!(session.observe(0, &current, 1, prediction + 1).is_err());
    session
        .observe(0, &current, 1, prediction)
        .expect("matching ticket");
    current[0] = 1;
    assert!(session.finish_time(&current).is_err());
    assert!(session.prediction(0, &current).is_err());
    let prediction = session
        .prediction(1, &current)
        .expect("second channel ticket");
    session
        .observe(1, &current, 2, prediction)
        .expect("second channel observation");
    current[1] = 2;
    session.finish_time(&current).expect("closed row");
    assert!(session.prediction(1, &[0, 0]).is_err());
    assert!(session.prediction(0, &[0, 0]).is_ok());
}

#[test]
fn universal_session_rejects_starting_or_skipping_channels() {
    let mut session = three_channel_session();
    let mut current = [0i64; 3];

    let error = session
        .prediction(1, &current)
        .expect_err("a time row must start at channel zero");
    assert!(error.to_string().contains("expected channel 0, got 1"));

    let prediction = session.prediction(0, &current).expect("channel zero");
    session
        .observe(0, &current, 17, prediction)
        .expect("close channel zero");
    current[0] = 17;

    let error = session
        .prediction(2, &current)
        .expect_err("channel one cannot be skipped");
    assert!(error.to_string().contains("expected channel 1, got 2"));
    let error = session
        .observe(2, &current, 9, prediction)
        .expect_err("an observation cannot skip channel one");
    assert!(error.to_string().contains("expected channel 1, got 2"));
}

#[test]
fn universal_session_rejects_duplicate_channel_after_observation_closes() {
    let mut session = three_channel_session();
    let mut current = [0i64; 3];
    let prediction = session.prediction(0, &current).expect("channel zero");
    session
        .observe(0, &current, 17, prediction)
        .expect("close channel zero");
    current[0] = 17;

    let error = session
        .prediction(0, &current)
        .expect_err("channel zero cannot be predicted twice");
    assert!(error.to_string().contains("expected channel 1, got 0"));
    let error = session
        .observe(0, &current, 17, prediction)
        .expect_err("channel zero cannot be observed twice");
    assert!(error.to_string().contains("expected channel 1, got 0"));
}

#[test]
fn universal_session_rejects_empty_and_partial_time_rows() {
    let mut session = three_channel_session();
    let mut current = [0i64; 3];

    let error = session
        .finish_time(&current)
        .expect_err("an empty time row cannot finish");
    assert!(error.to_string().contains("completed 0 of 3 channels"));

    let prediction = session.prediction(0, &current).expect("channel zero");
    session
        .observe(0, &current, 17, prediction)
        .expect("close channel zero");
    current[0] = 17;
    let error = session
        .finish_time(&current)
        .expect_err("a partial time row cannot finish");
    assert!(error.to_string().contains("completed 1 of 3 channels"));
}

#[test]
fn universal_session_resets_schedule_after_a_complete_time_row() {
    let mut session = three_channel_session();
    let mut current = [0i64; 3];
    for (channel, sample) in [17i64, -3, 9].into_iter().enumerate() {
        let prediction = session
            .prediction(channel, &current)
            .expect("ordered prediction");
        session
            .observe(channel, &current, sample, prediction)
            .expect("ordered observation");
        current[channel] = sample;
    }
    session.finish_time(&current).expect("complete row");

    let error = session
        .prediction(1, &[0, 0, 0])
        .expect_err("the next row must restart at channel zero");
    assert!(error.to_string().contains("expected channel 0, got 1"));
    session
        .prediction(0, &[0, 0, 0])
        .expect("channel zero starts the next row");
}

#[test]
fn universal_session_binds_prediction_observation_and_finished_row() {
    let graph = FixedUniversalGraph::new(vec![None]).expect("valid graph");
    let mut session = UniversalSession::new(graph, 16).expect("bounded session");
    let prediction = session.prediction(0, &[0]).expect("prediction");

    let before_mismatch = session.clone();
    let error = session
        .observe(0, &[1], 17, prediction)
        .expect_err("observation row must match its ticket");
    assert!(error.to_string().contains("prediction ticket"));
    assert_eq!(session, before_mismatch);

    session
        .observe(0, &[0], 17, prediction)
        .expect("matching observation");
    let before_finish = session.clone();
    let error = session
        .finish_time(&[18])
        .expect_err("finished row must match observed samples");
    assert!(error.to_string().contains("observed samples"));
    assert_eq!(session, before_finish);
    session.finish_time(&[17]).expect("matching finished row");
}

#[test]
fn universal_session_enforces_declared_bit_depth() {
    let graph = FixedUniversalGraph::new(vec![None]).expect("valid graph");
    let mut session = UniversalSession::new(graph, 8).expect("bounded session");

    assert!(session.prediction(0, &[128]).is_err());
    let prediction = session.prediction(0, &[0]).expect("prediction");
    assert!(session.observe(0, &[0], -129, prediction).is_err());
    session
        .observe(0, &[0], -128, prediction)
        .expect("lower bound is valid");
    session.finish_time(&[-128]).expect("bounded row");
}

fn three_channel_session() -> UniversalSession {
    let graph = FixedUniversalGraph::new(vec![None, Some(0), Some(1)]).expect("valid graph");
    UniversalSession::new(graph, 16).expect("bounded session")
}

fn hex_bytes(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap();
            let low = (pair[1] as char).to_digit(16).unwrap();
            ((high << 4) | low) as u8
        })
        .collect()
}

fn shared_fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/fixed_universal_v1.json"))
        .expect("shared fixed-universal conformance fixture is valid JSON")
}

fn json_i64(value: &Value, label: &str) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("fixture {label} is a signed integer"))
}

fn json_i128(value: &Value, label: &str) -> i128 {
    i128::from(json_i64(value, label))
}

fn json_u64(value: &Value, label: &str) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("fixture {label} is an unsigned integer"))
}

fn json_usize(value: &Value, label: &str) -> usize {
    usize::try_from(json_u64(value, label)).unwrap_or_else(|_| panic!("fixture {label} fits usize"))
}

fn json_u8(value: &Value, label: &str) -> u8 {
    u8::try_from(json_u64(value, label)).unwrap_or_else(|_| panic!("fixture {label} fits u8"))
}

fn json_i64_vec(value: &Value, label: &str) -> Vec<i64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("fixture {label} is an array"))
        .iter()
        .map(|item| json_i64(item, label))
        .collect()
}

fn json_i128_vec(value: &Value, label: &str) -> Vec<i128> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("fixture {label} is an array"))
        .iter()
        .map(|item| json_i128(item, label))
        .collect()
}

fn json_i128_matrix(value: &Value, label: &str) -> Vec<Vec<i128>> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("fixture {label} is a matrix"))
        .iter()
        .map(|row| json_i128_vec(row, label))
        .collect()
}

fn json_string_vec<'a>(value: &'a Value, label: &str) -> Vec<&'a str> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("fixture {label} is an array"))
        .iter()
        .map(|item| {
            item.as_str()
                .unwrap_or_else(|| panic!("fixture {label} contains only strings"))
        })
        .collect()
}

fn json_parents(value: &Value) -> Vec<Option<u16>> {
    value
        .as_array()
        .expect("fixture parents is an array")
        .iter()
        .map(|parent| {
            if parent.is_null() {
                None
            } else {
                Some(u16::try_from(json_u64(parent, "parent")).expect("fixture parent fits u16"))
            }
        })
        .collect()
}

fn assert_session_case(case: &Value) {
    let name = case["name"]
        .as_str()
        .expect("session case name is a string");
    let parents = json_parents(&case["parents"]);
    let samples = case["samples"]
        .as_array()
        .expect("session samples is an array")
        .iter()
        .map(|channel| json_i64_vec(channel, "session channel samples"))
        .collect::<Vec<_>>();
    assert_eq!(samples.len(), parents.len(), "{name}: channel count");
    let time_count = samples[0].len();
    assert!(
        samples.iter().all(|channel| channel.len() == time_count),
        "{name}: rectangular samples"
    );
    let graph = FixedUniversalGraph::new(parents).expect("fixture session graph is causal");
    let mut session =
        UniversalSession::new(graph, json_u8(&case["bit_depth"], "session bit_depth"))
            .expect("fixture session is bounded");
    let mut residuals = vec![Vec::with_capacity(time_count); samples.len()];

    for time in 0..time_count {
        let mut current = vec![0i64; samples.len()];
        for channel in 0..samples.len() {
            let prediction = session
                .prediction(channel, &current)
                .expect("fixture prediction is bounded");
            let sample = samples[channel][time];
            residuals[channel].push(sample - prediction);
            session
                .observe(channel, &current, sample, prediction)
                .expect("fixture observation is bounded");
            current[channel] = sample;
        }
        session
            .finish_time(&current)
            .expect("fixture time row is complete");
    }

    let encoded = residuals
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(encoded)),
        case["residual_i64le_channel_major_sha256"]
            .as_str()
            .expect("fixture residual hash is a string"),
        "{name}: residual hash"
    );
    assert_eq!(
        session.reset_count().expect("bounded session reset count"),
        json_u64(&case["reset_count"], "session reset_count"),
        "{name}: reset count"
    );
}
