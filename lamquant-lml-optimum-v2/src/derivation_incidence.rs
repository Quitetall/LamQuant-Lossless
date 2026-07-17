//! Collision-free derivation incidence for the wire-free DIX1 prototype.
//!
//! EEG channels are directed electrode edges. Earlier edges form a canonical
//! spanning forest; a later edge whose endpoints are already connected can be
//! predicted by the signed path through that forest. Endpoint strings are
//! namespaced and retained exactly after normalization rather than hashed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::OptimumV2Error;

pub const MAX_INCIDENCE_SUPPORTS: usize = 4;
const MAX_CHANNELS: usize = 256;
const MAX_LABEL_BYTES: usize = 255;
const MONOPOLAR_REFERENCE: &str = "R:MONO";

const REFERENCE_TOKENS: &[&str] = &["REF", "LE", "AR", "AVG", "CZREF"];
const AUX_PREFIXES: &[&str] = &[
    "ECG", "EKG", "RESP", "SPO2", "SP02", "PULSE", "HEART", "HR", "EMG", "EOG", "MARK", "MK",
    "TRIGGER", "EVENT", "TEMP", "CO2", "AIRFLOW",
];
const ELECTRODE_FAMILIES: &[&str] = &[
    "FP", "AF", "FC", "FT", "TP", "CP", "PO", "F", "T", "C", "P", "O", "I", "A", "M",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelIdentity {
    pub stable_id: u16,
    pub label: String,
}

impl ChannelIdentity {
    pub fn new(stable_id: u16, label: impl Into<String>) -> Self {
        Self {
            stable_id,
            label: label.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Partition {
    Eeg,
    Aux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DerivationKind {
    Monopolar,
    Referential,
    Bipolar,
    Aux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncidenceSupport {
    pub prior_channel: usize,
    pub coefficient: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncidenceChannel {
    canonical_index: usize,
    presented_index: usize,
    stable_id: u16,
    normalized_label: String,
    partition: Partition,
    kind: DerivationKind,
    positive_endpoint: Option<String>,
    negative_endpoint: Option<String>,
    supports: Vec<IncidenceSupport>,
}

impl IncidenceChannel {
    pub fn canonical_index(&self) -> usize {
        self.canonical_index
    }

    pub fn presented_index(&self) -> usize {
        self.presented_index
    }

    pub fn stable_id(&self) -> u16 {
        self.stable_id
    }

    pub fn normalized_label(&self) -> &str {
        &self.normalized_label
    }

    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn kind(&self) -> DerivationKind {
        self.kind
    }

    pub fn positive_endpoint(&self) -> Option<&str> {
        self.positive_endpoint.as_deref()
    }

    pub fn negative_endpoint(&self) -> Option<&str> {
        self.negative_endpoint.as_deref()
    }

    pub fn supports(&self) -> &[IncidenceSupport] {
        &self.supports
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationIncidence {
    channels: Vec<IncidenceChannel>,
    canonical_to_presented: Vec<usize>,
    presented_to_canonical: Vec<usize>,
}

impl DerivationIncidence {
    pub fn build(identities: &[ChannelIdentity]) -> Result<Self, OptimumV2Error> {
        if !(1..=MAX_CHANNELS).contains(&identities.len()) {
            return Err(invalid("DIX1 channel count is outside 1..=256"));
        }
        let mut stable_ids = BTreeSet::new();
        let mut parsed = Vec::with_capacity(identities.len());
        for (presented_index, identity) in identities.iter().enumerate() {
            if !stable_ids.insert(identity.stable_id) {
                return Err(invalid("DIX1 stable channel identities must be unique"));
            }
            parsed.push(parse_channel(identity, presented_index)?);
        }
        parsed.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));

        let canonical_to_presented: Vec<usize> = parsed
            .iter()
            .map(|channel| channel.presented_index)
            .collect();
        let mut presented_to_canonical = vec![0; parsed.len()];
        for (canonical, &presented) in canonical_to_presented.iter().enumerate() {
            presented_to_canonical[presented] = canonical;
        }

        let mut endpoints: BTreeMap<String, usize> = BTreeMap::new();
        let mut forest: Vec<Vec<ForestStep>> = Vec::new();
        let mut channels = Vec::with_capacity(parsed.len());
        for (canonical_index, channel) in parsed.into_iter().enumerate() {
            let supports = if channel.partition == Partition::Eeg {
                let positive = channel
                    .positive_endpoint
                    .as_deref()
                    .ok_or_else(|| invalid("DIX1 EEG channel lacks a positive endpoint"))?;
                let negative = channel
                    .negative_endpoint
                    .as_deref()
                    .ok_or_else(|| invalid("DIX1 EEG channel lacks a negative endpoint"))?;
                let positive_node = endpoint_node(positive, &mut endpoints, &mut forest);
                let negative_node = endpoint_node(negative, &mut endpoints, &mut forest);
                match signed_path(&forest, negative_node, positive_node) {
                    Some(path) if path.len() <= MAX_INCIDENCE_SUPPORTS => path,
                    Some(_) => Vec::new(),
                    None => {
                        forest[negative_node].push(ForestStep {
                            next: positive_node,
                            channel: canonical_index,
                            coefficient: 1,
                        });
                        forest[positive_node].push(ForestStep {
                            next: negative_node,
                            channel: canonical_index,
                            coefficient: -1,
                        });
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            if supports.iter().any(|support| {
                support.prior_channel >= canonical_index || !matches!(support.coefficient, -1 | 1)
            }) {
                return Err(invalid("DIX1 incidence support is not strictly causal"));
            }
            channels.push(IncidenceChannel {
                canonical_index,
                presented_index: channel.presented_index,
                stable_id: channel.stable_id,
                normalized_label: channel.normalized_label,
                partition: channel.partition,
                kind: channel.kind,
                positive_endpoint: channel.positive_endpoint,
                negative_endpoint: channel.negative_endpoint,
                supports,
            });
        }

        Ok(Self {
            channels,
            canonical_to_presented,
            presented_to_canonical,
        })
    }

    pub fn channels(&self) -> &[IncidenceChannel] {
        &self.channels
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn canonical_to_presented(&self) -> &[usize] {
        &self.canonical_to_presented
    }

    pub fn presented_to_canonical(&self) -> &[usize] {
        &self.presented_to_canonical
    }

    pub fn canonicalize_row(&self, presented: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        if presented.len() != self.channel_count() {
            return Err(invalid("DIX1 presented row has the wrong channel count"));
        }
        Ok(self
            .canonical_to_presented
            .iter()
            .map(|&index| presented[index])
            .collect())
    }

    pub fn restore_presented_row(&self, canonical: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        if canonical.len() != self.channel_count() {
            return Err(invalid("DIX1 canonical row has the wrong channel count"));
        }
        let mut presented = vec![0; canonical.len()];
        for (canonical_index, &presented_index) in self.canonical_to_presented.iter().enumerate() {
            presented[presented_index] = canonical[canonical_index];
        }
        Ok(presented)
    }
}

#[derive(Debug, Clone)]
struct ParsedChannel {
    presented_index: usize,
    stable_id: u16,
    normalized_label: String,
    partition: Partition,
    kind: DerivationKind,
    positive_endpoint: Option<String>,
    negative_endpoint: Option<String>,
}

impl ParsedChannel {
    fn sort_key(&self) -> (Partition, DerivationKind, &str, &str, &str, u16) {
        (
            self.partition,
            self.kind,
            self.positive_endpoint.as_deref().unwrap_or(""),
            self.negative_endpoint.as_deref().unwrap_or(""),
            &self.normalized_label,
            self.stable_id,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ForestStep {
    next: usize,
    channel: usize,
    coefficient: i8,
}

fn parse_channel(
    identity: &ChannelIdentity,
    presented_index: usize,
) -> Result<ParsedChannel, OptimumV2Error> {
    let normalized = normalize_label(&identity.label)?;
    if is_known_aux(&normalized) {
        return Ok(aux_channel(identity, presented_index, normalized));
    }
    let pieces: Vec<String> = normalized.split('-').map(str::to_owned).collect();
    match pieces.as_slice() {
        [positive] if is_electrode(positive) => Ok(ParsedChannel {
            presented_index,
            stable_id: identity.stable_id,
            normalized_label: normalized,
            partition: Partition::Eeg,
            kind: DerivationKind::Monopolar,
            positive_endpoint: Some(format!("E:{positive}")),
            negative_endpoint: Some(MONOPOLAR_REFERENCE.into()),
        }),
        [positive, negative] if is_electrode(positive) => {
            if REFERENCE_TOKENS.contains(&negative.as_str()) {
                Ok(ParsedChannel {
                    presented_index,
                    stable_id: identity.stable_id,
                    normalized_label: normalized,
                    partition: Partition::Eeg,
                    kind: DerivationKind::Referential,
                    positive_endpoint: Some(format!("E:{positive}")),
                    negative_endpoint: Some(format!("R:{negative}")),
                })
            } else if is_electrode(negative) {
                if positive == negative {
                    return Err(invalid("DIX1 bipolar endpoints must be distinct"));
                }
                Ok(ParsedChannel {
                    presented_index,
                    stable_id: identity.stable_id,
                    normalized_label: normalized,
                    partition: Partition::Eeg,
                    kind: DerivationKind::Bipolar,
                    positive_endpoint: Some(format!("E:{positive}")),
                    negative_endpoint: Some(format!("E:{negative}")),
                })
            } else {
                Ok(aux_channel(identity, presented_index, normalized))
            }
        }
        _ => Ok(aux_channel(identity, presented_index, normalized)),
    }
}

fn aux_channel(
    identity: &ChannelIdentity,
    presented_index: usize,
    normalized_label: String,
) -> ParsedChannel {
    ParsedChannel {
        presented_index,
        stable_id: identity.stable_id,
        normalized_label,
        partition: Partition::Aux,
        kind: DerivationKind::Aux,
        positive_endpoint: None,
        negative_endpoint: None,
    }
}

fn normalize_label(label: &str) -> Result<String, OptimumV2Error> {
    if label.is_empty() || label.contains('\0') || label.len() > MAX_LABEL_BYTES {
        return Err(invalid("DIX1 channel label is empty, unsafe, or too long"));
    }
    let mut normalized = label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_uppercase();
    if let Some(stripped) = normalized.strip_prefix("EEG ") {
        normalized = stripped.trim().to_owned();
    }
    while normalized.ends_with('.') {
        normalized.pop();
    }
    if normalized.is_empty() || normalized.len() > MAX_LABEL_BYTES {
        return Err(invalid("DIX1 channel label is empty after normalization"));
    }
    Ok(normalized)
}

fn is_known_aux(label: &str) -> bool {
    AUX_PREFIXES.iter().any(|prefix| {
        label == *prefix
            || label.starts_with(&format!("{prefix} "))
            || label.starts_with(&format!("{prefix}-"))
    })
}

fn is_electrode(token: &str) -> bool {
    ELECTRODE_FAMILIES.iter().any(|family| {
        token.strip_prefix(family).is_some_and(|suffix| {
            suffix == "Z"
                || (!suffix.is_empty()
                    && suffix.len() <= 2
                    && suffix.bytes().all(|byte| byte.is_ascii_digit())
                    && suffix.parse::<u8>().is_ok_and(|number| number > 0))
        })
    })
}

fn endpoint_node(
    endpoint: &str,
    endpoints: &mut BTreeMap<String, usize>,
    forest: &mut Vec<Vec<ForestStep>>,
) -> usize {
    if let Some(&node) = endpoints.get(endpoint) {
        return node;
    }
    let node = forest.len();
    endpoints.insert(endpoint.to_owned(), node);
    forest.push(Vec::new());
    node
}

fn signed_path(
    forest: &[Vec<ForestStep>],
    start: usize,
    target: usize,
) -> Option<Vec<IncidenceSupport>> {
    if start == target {
        return Some(Vec::new());
    }
    let mut previous: Vec<Option<(usize, usize, i8)>> = vec![None; forest.len()];
    let mut queue = VecDeque::from([start]);
    previous[start] = Some((start, usize::MAX, 0));
    while let Some(node) = queue.pop_front() {
        for step in &forest[node] {
            if previous[step.next].is_some() {
                continue;
            }
            previous[step.next] = Some((node, step.channel, step.coefficient));
            if step.next == target {
                queue.clear();
                break;
            }
            queue.push_back(step.next);
        }
    }
    previous[target]?;
    let mut path = Vec::new();
    let mut node = target;
    while node != start {
        let (parent, channel, coefficient) = previous[node]?;
        path.push(IncidenceSupport {
            prior_channel: channel,
            coefficient,
        });
        node = parent;
    }
    path.reverse();
    Some(path)
}

fn invalid(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}
