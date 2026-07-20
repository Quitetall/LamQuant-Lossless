//! Metadata-only construction gate for DIX2 TreeMED topology.

use lamquant_lml_optimum_v2::derivation_forest::{DerivationForest, TreeInnovationSession};
use lamquant_lml_optimum_v2::derivation_incidence::{
    ChannelIdentity, DerivationIncidence, Partition,
};

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

#[test]
fn common_reference_channels_form_a_causal_positive_forest() {
    let forest = DerivationForest::build(&[
        identity(10, "EEG FP1-REF"),
        identity(11, "F3-REF"),
        identity(12, "C3-REF"),
        identity(13, "ECG"),
    ])
    .expect("metadata-only forest");

    let eeg: Vec<_> = forest
        .channels()
        .iter()
        .filter(|channel| channel.partition() == Partition::Eeg)
        .collect();
    assert_eq!(eeg.len(), 3);
    assert!(eeg[0].parent().is_none());
    assert_eq!(eeg[1].supports().len(), 1);
    assert_eq!(eeg[2].supports().len(), 2);
    for channel in &eeg[1..] {
        let parent = channel.parent().expect("shared-reference parent");
        assert!(parent.parent_channel < channel.canonical_index());
        assert_eq!(parent.coefficient, 1);
        assert_eq!(parent.shared_endpoint, "R:REF");
    }
    let aux = forest
        .channels()
        .iter()
        .find(|channel| channel.partition() == Partition::Aux)
        .expect("AUX channel");
    assert!(aux.parent().is_none());
}

#[test]
fn tree_med_uses_three_causal_polarity_adjusted_innovations() {
    let identities = [
        identity(0, "C3-REF"),
        identity(1, "C4-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
    let forest = DerivationForest::build(&identities).expect("forest");
    assert_eq!(forest.channels()[3].supports().len(), 3);
    let mut session = TreeInnovationSession::new(&identities).expect("session");
    assert_eq!(session.forward_row(&[0, 0, 0, 0]).expect("origin"), [0; 4]);
    let residuals = session.forward_row(&[10, 100, 20, 20]).expect("median row");
    assert_eq!(residuals[3], 0, "median(10, 100, 20) predicts 20");
}

#[test]
fn bipolar_shared_electrodes_fix_parent_polarity() {
    let forest = DerivationForest::build(&[
        identity(0, "F3-C3"),
        identity(1, "C3-P3"),
        identity(2, "P3-O1"),
    ])
    .expect("bipolar forest");

    let f3_c3 = forest
        .channels()
        .iter()
        .find(|channel| channel.normalized_label() == "F3-C3")
        .expect("F3-C3");
    let support = f3_c3.parent().expect("shared C3 parent");
    assert_eq!(support.shared_endpoint, "E:C3");
    assert_eq!(support.coefficient, -1);

    let p3_o1 = forest
        .channels()
        .iter()
        .find(|channel| channel.normalized_label() == "P3-O1")
        .expect("P3-O1");
    let support = p3_o1.parent().expect("shared P3 parent");
    assert_eq!(support.shared_endpoint, "E:P3");
    assert_eq!(support.coefficient, -1);
}

#[test]
fn forest_is_permutation_equivariant_and_payload_free() {
    let first = vec![
        identity(7, "FP1-REF"),
        identity(2, "C3-P3"),
        identity(9, "F3-C3"),
        identity(4, "P3-O1"),
        identity(6, "ECG"),
    ];
    let second = vec![
        first[4].clone(),
        first[2].clone(),
        first[0].clone(),
        first[3].clone(),
        first[1].clone(),
    ];
    let left = DerivationForest::build(&first).expect("left forest");
    let right = DerivationForest::build(&second).expect("right forest");

    let semantics = |forest: &DerivationForest| {
        forest
            .channels()
            .iter()
            .map(|channel| {
                (
                    channel.stable_id(),
                    channel.normalized_label().to_owned(),
                    channel.supports().to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(semantics(&left), semantics(&right));
    assert!(left.channels().iter().all(|channel| channel
        .parent()
        .is_none_or(|parent| parent.parent_channel < channel.canonical_index())));
}

#[test]
fn tree_innovations_roundtrip_and_follow_identity_not_presentation_order() {
    let identities = vec![
        identity(7, "FP1-REF"),
        identity(2, "C3-REF"),
        identity(9, "F3-REF"),
        identity(6, "ECG"),
    ];
    let permutation = [3usize, 2, 0, 1];
    let permuted_identities = permutation
        .iter()
        .map(|&index| identities[index].clone())
        .collect::<Vec<_>>();
    let mut encoder = TreeInnovationSession::new(&identities).expect("encoder");
    let mut decoder = TreeInnovationSession::new(&identities).expect("decoder");
    let mut permuted = TreeInnovationSession::new(&permuted_identities).expect("permuted encoder");

    for row in common_reference_rows(257) {
        let residuals = encoder.forward_row(&row).expect("forward row");
        let permuted_row = permutation
            .iter()
            .map(|&index| row[index])
            .collect::<Vec<_>>();
        assert_eq!(
            residuals,
            permuted
                .forward_row(&permuted_row)
                .expect("permuted forward")
        );
        assert_eq!(decoder.inverse_row(&residuals).expect("inverse row"), row);
    }
    assert_eq!(encoder.row_count(), 257);
    assert_eq!(encoder, decoder);
}

#[test]
fn tree_innovations_reduce_common_reference_varint_proxy() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "ECG"),
    ];
    let rows = common_reference_rows(512);
    let mut tree = TreeInnovationSession::new(&identities).expect("tree");
    let tree_bytes: usize = rows
        .iter()
        .map(|row| {
            tree.forward_row(row)
                .expect("tree row")
                .iter()
                .map(|&value| signed_varint_len(value))
                .sum::<usize>()
        })
        .sum();
    let incidence = DerivationIncidence::build(&identities).expect("canonical order");
    let mut previous = vec![0i64; identities.len()];
    let mut temporal_bytes = 0usize;
    for row in &rows {
        let canonical = incidence.canonicalize_row(row).expect("canonical row");
        for (channel, &sample) in canonical.iter().enumerate() {
            temporal_bytes += signed_varint_len(sample - previous[channel]);
            previous[channel] = sample;
        }
    }
    assert!(
        tree_bytes * 100 <= temporal_bytes * 80,
        "TreeMED innovation proxy must save at least 20% on shared-reference synthetic data: {tree_bytes} vs {temporal_bytes}"
    );
}

#[test]
fn tree_innovations_fail_on_arithmetic_overflow() {
    let identities = [identity(0, "C3-REF")];
    let mut session = TreeInnovationSession::new(&identities).expect("session");
    session.forward_row(&[i64::MAX]).expect("first row");
    assert!(session.forward_row(&[i64::MIN]).is_err());
}

fn common_reference_rows(count: usize) -> Vec<Vec<i64>> {
    (0..count)
        .map(|time| {
            let common = ((time as i64 * 97) % 4096) - 2048;
            vec![
                common + (time as i64 % 5),
                common - (time as i64 % 7),
                common + (time as i64 % 3),
                if time % 2 == 0 { 30_000 } else { -30_000 },
            ]
        })
        .collect()
}

fn signed_varint_len(value: i64) -> usize {
    let mut zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut bytes = 1;
    while zigzag >= 0x80 {
        zigzag >>= 7;
        bytes += 1;
    }
    bytes
}
