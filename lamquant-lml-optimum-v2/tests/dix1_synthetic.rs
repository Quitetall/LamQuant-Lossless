//! Synthetic construction gate for the wire-free DIX1 structural prototype.

use lamquant_lml_optimum_v2::derivation_incidence::{
    ChannelIdentity, DerivationIncidence, Partition,
};
use lamquant_lml_optimum_v2::dix1::Dix1Session;

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

#[test]
fn incidence_builds_exact_signed_cycles_and_isolates_aux() {
    let incidence = DerivationIncidence::build(&[
        identity(10, "EEG F3-REF"),
        identity(11, "C3-REF"),
        identity(12, "F3-C3"),
        identity(13, "ECG"),
    ])
    .expect("bounded incidence");

    let bipolar = incidence
        .channels()
        .iter()
        .find(|channel| channel.normalized_label() == "F3-C3")
        .expect("bipolar channel");
    assert_eq!(bipolar.partition(), Partition::Eeg);
    assert_eq!(bipolar.supports().len(), 2);
    assert!(bipolar
        .supports()
        .iter()
        .all(|support| support.prior_channel < bipolar.canonical_index()));

    let aux = incidence
        .channels()
        .iter()
        .find(|channel| channel.normalized_label() == "ECG")
        .expect("AUX channel");
    assert_eq!(aux.partition(), Partition::Aux);
    assert!(aux.supports().is_empty());
    assert!(aux.positive_endpoint().is_none());
    assert!(aux.negative_endpoint().is_none());

    let canonical = incidence
        .canonicalize_row(&[101, 37, 64, -9])
        .expect("canonical row");
    let predicted: i64 = bipolar
        .supports()
        .iter()
        .map(|support| i64::from(support.coefficient) * canonical[support.prior_channel])
        .sum();
    assert_eq!(predicted, 64);
}

#[test]
fn incidence_is_permutation_equivariant_and_duplicate_edges_are_exact() {
    let first = vec![
        identity(4, "F3-REF"),
        identity(1, "C3-REF"),
        identity(7, "F3-C3"),
        identity(9, "F3-REF"),
        identity(8, "mystery sensor"),
    ];
    let permutation = [4usize, 2, 0, 3, 1];
    let second: Vec<_> = permutation
        .iter()
        .map(|&index| first[index].clone())
        .collect();
    let left = DerivationIncidence::build(&first).expect("left incidence");
    let right = DerivationIncidence::build(&second).expect("right incidence");

    let left_semantics: Vec<_> = left
        .channels()
        .iter()
        .map(|channel| {
            (
                channel.stable_id(),
                channel.normalized_label().to_owned(),
                channel.partition(),
                channel.supports().to_vec(),
            )
        })
        .collect();
    let right_semantics: Vec<_> = right
        .channels()
        .iter()
        .map(|channel| {
            (
                channel.stable_id(),
                channel.normalized_label().to_owned(),
                channel.partition(),
                channel.supports().to_vec(),
            )
        })
        .collect();
    assert_eq!(left_semantics, right_semantics);

    let duplicate = left
        .channels()
        .iter()
        .find(|channel| channel.stable_id() == 9)
        .expect("second duplicate derivation");
    assert_eq!(duplicate.supports().len(), 1);
    assert_eq!(duplicate.supports()[0].coefficient, 1);
    assert!(duplicate.supports()[0].prior_channel < duplicate.canonical_index());

    let unknown = left
        .channels()
        .iter()
        .find(|channel| channel.stable_id() == 8)
        .expect("unknown sensor");
    assert_eq!(unknown.partition(), Partition::Aux);
    assert!(unknown.supports().is_empty());
}

#[test]
fn incidence_omits_paths_beyond_the_four_support_bound() {
    let incidence = DerivationIncidence::build(&[
        identity(0, "C1-F1"),
        identity(1, "C2-C1"),
        identity(2, "C3-C2"),
        identity(3, "C4-C3"),
        identity(4, "C5-C4"),
        identity(5, "F1-C5"),
    ])
    .expect("bounded chain");
    let long_chord = incidence
        .channels()
        .iter()
        .find(|channel| channel.normalized_label() == "F1-C5")
        .expect("long chord");
    assert!(long_chord.supports().is_empty());
}

#[test]
fn incidence_rejects_ambiguous_identity_and_unsafe_labels() {
    assert!(DerivationIncidence::build(&[identity(2, "F3-REF"), identity(2, "C3-REF")]).is_err());
    assert!(DerivationIncidence::build(&[identity(0, "   ")]).is_err());
    assert!(DerivationIncidence::build(&[identity(0, "F3\0REF")]).is_err());
    assert!(DerivationIncidence::build(&[identity(0, "F3-F3")]).is_err());
    assert!(DerivationIncidence::build(&[]).is_err());
}

#[test]
fn dix1_roundtrips_permuted_montage_with_identical_state_and_residual_proxy_bytes() {
    let identities = vec![
        identity(10, "F3-REF"),
        identity(11, "C3-REF"),
        identity(12, "F3-C3"),
        identity(13, "ECG"),
    ];
    let permutation = [3usize, 2, 0, 1];
    let permuted_identities: Vec<_> = permutation
        .iter()
        .map(|&index| identities[index].clone())
        .collect();
    let rows = incidence_fixture(512);

    let mut encoder = Dix1Session::new(&identities, 24, 500_000).expect("encoder");
    let mut repeat = Dix1Session::new(&identities, 24, 500_000).expect("repeat");
    let mut permuted =
        Dix1Session::new(&permuted_identities, 24, 500_000).expect("permuted encoder");
    let mut decoder = Dix1Session::new(&identities, 24, 500_000).expect("decoder");
    assert_eq!(encoder.sample_lags(), [1, 2, 8, 32]);

    let mut encoded = Vec::new();
    let mut repeated = Vec::new();
    for row in &rows {
        let residuals = encoder.forward_row(row).expect("DIX1 forward");
        let repeat_residuals = repeat.forward_row(row).expect("repeat forward");
        assert_eq!(residuals, repeat_residuals);
        push_signed_varints(&mut encoded, &residuals);
        push_signed_varints(&mut repeated, &repeat_residuals);

        let permuted_row: Vec<_> = permutation.iter().map(|&index| row[index]).collect();
        let permuted_residuals = permuted
            .forward_row(&permuted_row)
            .expect("permuted forward");
        assert_eq!(residuals, permuted_residuals);

        let decoded = decoder.inverse_row(&residuals).expect("DIX1 inverse");
        assert_eq!(&decoded, row);
        assert_eq!(encoder.state_digest(), decoder.state_digest());
        assert_eq!(encoder.state_digest(), permuted.state_digest());
    }
    assert_eq!(encoded, repeated);
    assert_eq!(encoder, repeat);
    assert_eq!(encoder.row_count(), rows.len() as u64);

    let no_incidence_identities = vec![
        identity(10, "F3-REF"),
        identity(11, "C3-REF"),
        identity(12, "P3-REF"),
        identity(13, "ECG"),
    ];
    let no_incidence = DerivationIncidence::build(&no_incidence_identities)
        .expect("no-incidence control topology");
    let candidate_order: Vec<_> = encoder
        .incidence()
        .channels()
        .iter()
        .map(|channel| channel.stable_id())
        .collect();
    let control_order: Vec<_> = no_incidence
        .channels()
        .iter()
        .map(|channel| channel.stable_id())
        .collect();
    assert_eq!(candidate_order, control_order);
    assert!(no_incidence
        .channels()
        .iter()
        .all(|channel| channel.supports().is_empty()));
    let no_incidence_bytes = dix1_varint_bytes(&no_incidence_identities, &rows);
    let delta_bytes = delta_varint_bytes(&identities, &rows);
    println!(
        "DIX1 residual-payload varint proxy: incidence={} no_incidence={} incidence_saving={:.6}% delta={} delta_saving={:.6}%",
        encoded.len(),
        no_incidence_bytes.len(),
        100.0 * (no_incidence_bytes.len() - encoded.len()) as f64
            / no_incidence_bytes.len() as f64,
        delta_bytes.len(),
        100.0 * (delta_bytes.len() - encoded.len()) as f64 / delta_bytes.len() as f64
    );
    assert!(
        encoded.len() * 100 <= no_incidence_bytes.len() * 95,
        "DIX1 incidence arm must beat the same predictor with its topology broken by at least 5%: {} vs {}",
        encoded.len(),
        no_incidence_bytes.len()
    );
}

#[test]
fn dix1_aux_perturbations_cannot_change_eeg_residuals() {
    let identities = vec![
        identity(10, "F3-REF"),
        identity(11, "C3-REF"),
        identity(12, "F3-C3"),
        identity(13, "ECG"),
    ];
    let rows = incidence_fixture(256);
    let mut baseline = Dix1Session::new(&identities, 24, 500_000).expect("baseline");
    let mut perturbed = Dix1Session::new(&identities, 24, 500_000).expect("perturbed");
    let eeg_indices: Vec<_> = baseline
        .incidence()
        .channels()
        .iter()
        .enumerate()
        .filter_map(|(index, channel)| (channel.partition() == Partition::Eeg).then_some(index))
        .collect();

    for (time, row) in rows.iter().enumerate() {
        let mut changed = row.clone();
        changed[3] = if time % 2 == 0 { 7_000_000 } else { -7_000_000 };
        let expected = baseline.forward_row(row).expect("baseline forward");
        let actual = perturbed.forward_row(&changed).expect("perturbed forward");
        for &index in &eeg_indices {
            assert_eq!(actual[index], expected[index]);
        }
    }
}

#[test]
fn dix1_handles_nonstationarity_extremes_and_negative_control() {
    let identities = vec![identity(0, "F3"), identity(1, "C3"), identity(2, "ECG")];
    let rows = independent_fixture(512);
    let mut encoder = Dix1Session::new(&identities, 16, 256_000).expect("encoder");
    let mut decoder = Dix1Session::new(&identities, 16, 256_000).expect("decoder");
    assert_eq!(encoder.sample_lags(), [1, 1, 4, 16]);
    let mut bytes = Vec::new();
    for row in &rows {
        let residuals = encoder.forward_row(row).expect("negative-control forward");
        push_signed_varints(&mut bytes, &residuals);
        assert_eq!(decoder.inverse_row(&residuals).expect("inverse"), *row);
    }
    let delta_bytes = delta_varint_bytes(&identities, &rows);
    println!(
        "DIX1 negative control: candidate={} delta={} delta_pct={:.6}%",
        bytes.len(),
        delta_bytes.len(),
        100.0 * (bytes.len() as f64 / delta_bytes.len() as f64 - 1.0)
    );
    assert!(
        bytes.len() * 100 <= delta_bytes.len() * 101,
        "independent/AUX control regressed more than 1%: {} vs {}",
        bytes.len(),
        delta_bytes.len()
    );

    let extremes = vec![
        identity(0, "F3-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-C3"),
    ];
    let extreme_rows = [
        vec![i32::MAX as i64, 0, i32::MAX as i64],
        vec![i32::MIN as i64, -1, i32::MIN as i64 + 1],
        vec![0, i32::MAX as i64, -(i32::MAX as i64)],
        vec![-1, i32::MIN as i64, i32::MAX as i64],
    ];
    let mut extreme_encoder = Dix1Session::new(&extremes, 32, 250_000).expect("extreme encoder");
    let mut extreme_decoder = Dix1Session::new(&extremes, 32, 250_000).expect("extreme decoder");
    for row in extreme_rows {
        let residuals = extreme_encoder.forward_row(&row).expect("extreme forward");
        assert_eq!(
            extreme_decoder
                .inverse_row(&residuals)
                .expect("extreme inverse"),
            row
        );
    }
}

#[test]
fn dix1_failures_are_transactional_and_fail_closed() {
    let identities = [identity(0, "F3-REF"), identity(1, "C3-REF")];
    let mut session = Dix1Session::new(&identities, 16, 250_000).expect("session");
    let initial = session.state_digest();
    assert!(session.inverse_row(&[1]).is_err());
    assert_eq!(session.state_digest(), initial);
    assert!(session.inverse_row(&[i64::MAX, 0]).is_err());
    assert_eq!(session.state_digest(), initial);
    assert!(session.inverse_row(&[0, i64::MAX]).is_err());
    assert_eq!(session.state_digest(), initial);
    assert!(session.forward_row(&[0]).is_err());
    assert_eq!(session.state_digest(), initial);
    assert!(session.forward_row(&[32_768, 0]).is_err());
    assert_eq!(session.state_digest(), initial);
    assert!(Dix1Session::new(&identities, 0, 250_000).is_err());
    assert!(Dix1Session::new(&identities, 33, 250_000).is_err());
    assert!(Dix1Session::new(&identities, 16, 0).is_err());
    assert!(Dix1Session::new(&identities, 16, 4_000_001).is_err());
}

fn incidence_fixture(count: usize) -> Vec<Vec<i64>> {
    let mut state = 0x8b8b_8b8b_1234_5678u64;
    (0..count)
        .map(|time| {
            let common = bounded_noise(&mut state, 180_000);
            let drift = if time < count / 2 {
                time as i64 * 73
            } else {
                -(time as i64) * 91
            };
            let f3 = common + bounded_noise(&mut state, 70_000) + drift;
            let c3 = common + bounded_noise(&mut state, 70_000) - drift / 2;
            let ecg = bounded_noise(&mut state, 300_000);
            vec![f3, c3, f3 - c3, ecg]
        })
        .collect()
}

fn independent_fixture(count: usize) -> Vec<Vec<i64>> {
    let mut state = 0x1234_5678_9abc_def0u64;
    (0..count)
        .map(|time| {
            let sign = if time < count / 2 { 1 } else { -1 };
            vec![
                sign * bounded_noise(&mut state, 28_000),
                bounded_noise(&mut state, 28_000),
                bounded_noise(&mut state, 28_000),
            ]
        })
        .collect()
}

fn bounded_noise(state: &mut u64, bound: i64) -> i64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let span = (2 * bound + 1) as u64;
    (*state % span) as i64 - bound
}

fn delta_varint_bytes(identities: &[ChannelIdentity], rows: &[Vec<i64>]) -> Vec<u8> {
    let incidence = DerivationIncidence::build(identities).expect("delta incidence");
    let mut previous = vec![0; identities.len()];
    let mut bytes = Vec::new();
    for row in rows {
        let canonical = incidence
            .canonicalize_row(row)
            .expect("canonical delta row");
        let residuals: Vec<_> = canonical
            .iter()
            .zip(&previous)
            .map(|(&sample, &prior)| sample - prior)
            .collect();
        push_signed_varints(&mut bytes, &residuals);
        previous = canonical;
    }
    bytes
}

fn dix1_varint_bytes(identities: &[ChannelIdentity], rows: &[Vec<i64>]) -> Vec<u8> {
    let mut session = Dix1Session::new(identities, 24, 500_000).expect("DIX1 proxy session");
    let mut bytes = Vec::new();
    for row in rows {
        let residuals = session.forward_row(row).expect("DIX1 proxy forward");
        push_signed_varints(&mut bytes, &residuals);
    }
    bytes
}

fn push_signed_varints(bytes: &mut Vec<u8>, values: &[i64]) {
    for &value in values {
        let magnitude = value.unsigned_abs();
        let mut zigzag = if value >= 0 {
            magnitude * 2
        } else {
            magnitude * 2 - 1
        };
        loop {
            let mut byte = (zigzag & 0x7f) as u8;
            zigzag >>= 7;
            if zigzag != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if zigzag == 0 {
                break;
            }
        }
    }
}
