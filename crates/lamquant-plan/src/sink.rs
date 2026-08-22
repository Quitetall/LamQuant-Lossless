//! Bounded projection delivery and pollable plan-run snapshots.
//!
//! RELOCATED 2026-08-22 from `crates/lamquant-ops/src/sink.rs` in the PRIVATE
//! LamQuant meta-repository, byte-for-byte apart from re-pointing fourteen
//! `lamquant_plan::` paths at `crate::` -- this module now IS lamquant_plan.
//!
//! WHY IT BELONGS HERE. `lamquant-ops/src/lib.rs` states the rule the original
//! split followed: "what went is pure data; what stayed spawns processes and
//! opens sockets." Measured against that rule this file was on the wrong side.
//! Its entire import surface is `std::sync::{mpsc, Mutex}`,
//! `std::time::{Duration, Instant}`, `serde`, and the projection vocabulary
//! already defined in this crate. It spawns nothing and opens nothing: an mpsc
//! channel with a bounded queue, and a snapshot reducer that folds
//! `PlanUpdate`s and enforces identity stability. That is vocabulary for
//! DELIVERING projections, not machinery for producing them.
//!
//! WHAT THIS UNBLOCKS. A public front-end could name `PlanProjection` but had
//! no public way to RECEIVE one, so any consumer of the projection stream had
//! to reach into the private meta for `bounded_channel`. That was one of the
//! reasons a public repository could not be built from outside the owning
//! account (ADR 0185).
//!
//! `lamquant-ops` keeps `pub use lamquant_plan::sink;` plus its original
//! re-export list, so every existing `lamquant_ops::PlanRunState` and
//! `crate::sink::bounded_channel` call site resolves unchanged -- and resolves
//! to THIS definition, so no duplicate type can form at the seam.

use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{DiagnosticLevel, PlanIdentity, PlanProjection, PlanUpdate};

const MAX_SNAPSHOT_DIAGNOSTICS: usize = 1024;
const MAX_SNAPSHOT_DIAGNOSTIC_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlanRunState {
    Pending,
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
}

impl PlanRunState {
    pub const fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedDiagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRunSnapshot {
    pub operation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<PlanIdentity>,
    pub state: PlanRunState,
    pub completed_nodes: u32,
    pub total_nodes: u32,
    pub current: u64,
    pub total: u64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_artifact: Option<crate::ArtifactProjection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic_level: Option<DiagnosticLevel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<CapturedDiagnostic>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub diagnostics_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<crate::ExecutionReceiptProjection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<crate::FailureProjection>,
    pub updated_ms: i64,
}

const fn is_false(value: &bool) -> bool {
    !*value
}

impl PlanRunSnapshot {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            plan: None,
            state: PlanRunState::Pending,
            completed_nodes: 0,
            total_nodes: 0,
            current: 0,
            total: 0,
            message: String::new(),
            last_artifact: None,
            diagnostic_level: None,
            diagnostics: Vec::new(),
            diagnostics_truncated: false,
            terminal_message: None,
            receipt: None,
            failure: None,
            updated_ms: PlanProjection::now_ms(),
        }
    }

    pub fn mark_cancelling(&mut self) {
        if !self.state.terminal() {
            self.state = PlanRunState::Cancelling;
            self.updated_ms = PlanProjection::now_ms();
        }
    }

    pub fn apply(&mut self, projection: &PlanProjection) -> Result<(), String> {
        projection.validate()?;
        if projection.is_terminal() && !projection.terminal_is_executor_issued() {
            return Err("wire terminal lacks executor authority".into());
        }
        if self.state.terminal() {
            return Err("projection arrived after terminal plan state".into());
        }
        let is_planned = matches!(projection.update, PlanUpdate::Planned { .. });
        if self.plan.is_none() && !is_planned {
            return Err("first projection for a plan run must be planned".into());
        }
        if self.plan.is_some() && is_planned {
            return Err("plan run received duplicate planned projection".into());
        }
        if let Some(identity) = &self.plan {
            if identity != &projection.plan {
                return Err("projection identity changed within one plan run".into());
            }
        } else {
            self.plan = Some(projection.plan.clone());
        }
        self.updated_ms = projection.observed_at_ms;
        match &projection.update {
            PlanUpdate::Planned {
                operation,
                total_nodes,
                total_work,
            } => {
                self.operation = operation.clone();
                self.diagnostic_level = None;
                if self.state != PlanRunState::Cancelling {
                    self.state = PlanRunState::Running;
                }
                self.total_nodes = *total_nodes;
                if let Some(total) = total_work {
                    self.total = *total;
                }
            }
            PlanUpdate::Progress {
                node_id,
                current,
                total,
                message,
            } => {
                if self.state != PlanRunState::Cancelling {
                    self.state = PlanRunState::Running;
                }
                if *node_id >= self.total_nodes {
                    return Err("progress projection names node outside compiled plan".into());
                }
                self.current = *current;
                self.total = *total;
                self.message = message.clone();
                self.diagnostic_level = None;
            }
            PlanUpdate::Artifact { node_id, artifact } => {
                if *node_id >= self.total_nodes {
                    return Err("artifact projection names node outside compiled plan".into());
                }
                self.last_artifact = Some(artifact.clone());
                self.message = if artifact.success {
                    artifact.path.clone()
                } else {
                    format!("{} failed", artifact.path)
                };
                self.diagnostic_level = None;
            }
            PlanUpdate::Receipt {
                receipt, message, ..
            } => {
                self.validate_receipt_nodes(receipt)?;
                self.state = PlanRunState::Completed;
                self.completed_nodes = receipt.completed_node_ids.len() as u32;
                self.terminal_message = Some(message.clone());
                self.receipt = Some(receipt.clone());
                self.failure = None;
            }
            PlanUpdate::Failure {
                receipt,
                failure,
                cancelled,
                ..
            } => {
                self.validate_receipt_nodes(receipt)?;
                self.state = if *cancelled {
                    PlanRunState::Cancelled
                } else {
                    PlanRunState::Failed
                };
                self.completed_nodes = receipt.completed_node_ids.len() as u32;
                self.terminal_message = Some(failure.message.clone());
                self.receipt = Some(receipt.clone());
                self.failure = Some(failure.clone());
            }
            PlanUpdate::Diagnostic { level, message, .. } => {
                self.message = message.clone();
                self.diagnostic_level = Some(*level);
                self.capture_diagnostic(*level, message);
            }
        }
        Ok(())
    }

    fn capture_diagnostic(&mut self, level: DiagnosticLevel, message: &str) {
        let mut message = message.to_owned();
        if message.len() > MAX_SNAPSHOT_DIAGNOSTIC_BYTES {
            let mut end = MAX_SNAPSHOT_DIAGNOSTIC_BYTES;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
            self.diagnostics_truncated = true;
        }
        self.diagnostics.push(CapturedDiagnostic { level, message });
        let mut observed = self
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.len())
            .sum::<usize>();
        while self.diagnostics.len() > MAX_SNAPSHOT_DIAGNOSTICS
            || observed > MAX_SNAPSHOT_DIAGNOSTIC_BYTES
        {
            let removed = self.diagnostics.remove(0);
            observed = observed.saturating_sub(removed.message.len());
            self.diagnostics_truncated = true;
        }
    }

    fn validate_receipt_nodes(
        &self,
        receipt: &crate::ExecutionReceiptProjection,
    ) -> Result<(), String> {
        if receipt
            .completed_node_ids
            .iter()
            .chain(
                receipt
                    .attempts
                    .iter()
                    .flat_map(|attempt| attempt.node_ids.iter()),
            )
            .any(|node_id| *node_id >= self.total_nodes)
        {
            return Err("terminal receipt names node outside compiled plan".into());
        }
        Ok(())
    }
}

pub trait PlanProjectionSink: Send + Sync + 'static {
    fn project(&self, projection: PlanProjection);
}

pub struct MpscProjectionSink {
    send: std::sync::Mutex<ProjectionSendState>,
}

struct ProjectionSendState {
    observations: mpsc::SyncSender<ProjectionEnvelope>,
    lifecycle: mpsc::SyncSender<ProjectionEnvelope>,
    next_sequence: u64,
    plan: Option<crate::PlanIdentity>,
    planned: bool,
    terminal: bool,
}

struct ProjectionEnvelope {
    sequence: u64,
    projection: PlanProjection,
}

impl MpscProjectionSink {
    fn new_bounded(
        tx: mpsc::SyncSender<ProjectionEnvelope>,
        lifecycle_tx: mpsc::SyncSender<ProjectionEnvelope>,
    ) -> Self {
        Self {
            send: std::sync::Mutex::new(ProjectionSendState {
                observations: tx,
                lifecycle: lifecycle_tx,
                next_sequence: 0,
                plan: None,
                planned: false,
                terminal: false,
            }),
        }
    }
}

impl PlanProjectionSink for MpscProjectionSink {
    fn project(&self, projection: PlanProjection) {
        if let Err(error) = projection.validate() {
            eprintln!("WARNING: rejected invalid plan projection: {error}");
            return;
        }
        if projection.is_terminal() && !projection.terminal_is_executor_issued() {
            eprintln!("WARNING: rejected wire terminal without executor authority");
            return;
        }
        let mut send = match self.send.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let is_planned = matches!(projection.update, PlanUpdate::Planned { .. });
        let is_terminal = projection.is_terminal();
        if let Some(plan) = &send.plan {
            if plan != &projection.plan {
                eprintln!("WARNING: rejected plan projection with identity drift");
                return;
            }
        } else if !is_planned {
            eprintln!("WARNING: rejected plan projection before planned lifecycle record");
            return;
        }
        if is_planned {
            if send.planned {
                eprintln!("WARNING: rejected duplicate planned lifecycle record");
                return;
            }
            send.plan = Some(projection.plan.clone());
            send.planned = true;
        } else if send.terminal {
            eprintln!("WARNING: rejected plan projection after terminal lifecycle record");
            return;
        }
        if is_terminal {
            send.terminal = true;
        }
        let sequence = send.next_sequence;
        send.next_sequence = send.next_sequence.saturating_add(1);
        let lifecycle = matches!(
            projection.update,
            PlanUpdate::Planned { .. } | PlanUpdate::Receipt { .. } | PlanUpdate::Failure { .. }
        );
        let envelope = ProjectionEnvelope {
            sequence,
            projection,
        };
        if lifecycle {
            if matches!(
                send.lifecycle.try_send(envelope),
                Err(mpsc::TrySendError::Full(_))
            ) {
                eprintln!(
                    "WARNING: plan lifecycle channel full; rejecting excess lifecycle record"
                );
            }
            return;
        }
        if matches!(
            send.observations.try_send(envelope),
            Err(mpsc::TrySendError::Full(_))
        ) {
            eprintln!("WARNING: plan projection channel full; dropping observation");
        }
    }
}

pub struct ProjectionReceiver {
    state: Mutex<ProjectionReceiverState>,
}

struct ProjectionReceiverState {
    observations: mpsc::Receiver<ProjectionEnvelope>,
    lifecycle: mpsc::Receiver<ProjectionEnvelope>,
    pending_observation: Option<ProjectionEnvelope>,
    pending_lifecycle: Option<ProjectionEnvelope>,
    observations_closed: bool,
    lifecycle_closed: bool,
}

impl ProjectionReceiver {
    pub fn recv(&self) -> Result<PlanProjection, mpsc::RecvError> {
        loop {
            match self.try_recv() {
                Ok(projection) => return Ok(projection),
                Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvError),
                Err(mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    pub fn try_recv(&self) -> Result<PlanProjection, mpsc::TryRecvError> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.try_recv()
    }

    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<PlanProjection, mpsc::RecvTimeoutError> {
        let started = Instant::now();
        loop {
            match self.try_recv() {
                Ok(projection) => return Ok(projection),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(mpsc::RecvTimeoutError::Disconnected);
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if started.elapsed() >= timeout {
                return Err(mpsc::RecvTimeoutError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl ProjectionReceiverState {
    fn try_recv(&mut self) -> Result<PlanProjection, mpsc::TryRecvError> {
        if self.pending_observation.is_none() && !self.observations_closed {
            match self.observations.try_recv() {
                Ok(envelope) => self.pending_observation = Some(envelope),
                Err(mpsc::TryRecvError::Disconnected) => self.observations_closed = true,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.pending_lifecycle.is_none() && !self.lifecycle_closed {
            match self.lifecycle.try_recv() {
                Ok(envelope) => self.pending_lifecycle = Some(envelope),
                Err(mpsc::TryRecvError::Disconnected) => self.lifecycle_closed = true,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let take_lifecycle = match (&self.pending_observation, &self.pending_lifecycle) {
            (Some(observation), Some(lifecycle)) => lifecycle.sequence < observation.sequence,
            (None, Some(_)) => true,
            _ => false,
        };
        if take_lifecycle {
            return Ok(self
                .pending_lifecycle
                .take()
                .expect("lifecycle pending was checked")
                .projection);
        }
        if let Some(observation) = self.pending_observation.take() {
            return Ok(observation.projection);
        }
        if self.observations_closed && self.lifecycle_closed {
            Err(mpsc::TryRecvError::Disconnected)
        } else {
            Err(mpsc::TryRecvError::Empty)
        }
    }
}

pub const DEFAULT_CHANNEL_BOUND: usize = 16_384;
const LIFECYCLE_CHANNEL_BOUND: usize = 2;

pub fn bounded_channel() -> (MpscProjectionSink, ProjectionReceiver) {
    let (tx, observations) = mpsc::sync_channel(DEFAULT_CHANNEL_BOUND);
    let (lifecycle_tx, lifecycle) = mpsc::sync_channel(LIFECYCLE_CHANNEL_BOUND);
    (
        MpscProjectionSink::new_bounded(tx, lifecycle_tx),
        ProjectionReceiver {
            state: Mutex::new(ProjectionReceiverState {
                observations,
                lifecycle,
                pending_observation: None,
                pending_lifecycle: None,
                observations_closed: false,
                lifecycle_closed: false,
            }),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiagnosticLevel, PlanUpdate};

    fn identity() -> PlanIdentity {
        PlanIdentity::new(&[1; 32], &[2; 32], &[3; 32])
    }

    #[test]
    fn snapshot_rejects_identity_drift() {
        let mut snapshot = PlanRunSnapshot::new("encode_lma");
        snapshot
            .apply(&PlanProjection::new(
                identity(),
                PlanUpdate::Planned {
                    operation: "encode_lma".into(),
                    total_nodes: 1,
                    total_work: None,
                },
            ))
            .unwrap();
        let mut other = identity();
        other.plan_id = "ff".repeat(32);
        assert!(snapshot
            .apply(&PlanProjection::new(
                other,
                PlanUpdate::Diagnostic {
                    node_id: Some(0),
                    level: DiagnosticLevel::Info,
                    message: "wrong plan".into(),
                },
            ))
            .is_err());
    }

    #[test]
    fn snapshot_retains_bounded_diagnostic_history() {
        let mut snapshot = PlanRunSnapshot::new("info");
        snapshot
            .apply(&PlanProjection::new(
                identity(),
                PlanUpdate::Planned {
                    operation: "info".into(),
                    total_nodes: 1,
                    total_work: None,
                },
            ))
            .unwrap();
        for index in 0..(MAX_SNAPSHOT_DIAGNOSTICS + 10) {
            snapshot
                .apply(&PlanProjection::new(
                    identity(),
                    PlanUpdate::Diagnostic {
                        node_id: Some(0),
                        level: DiagnosticLevel::Info,
                        message: format!("diagnostic-{index}"),
                    },
                ))
                .unwrap();
        }
        assert_eq!(snapshot.diagnostics.len(), MAX_SNAPSHOT_DIAGNOSTICS);
        assert!(snapshot.diagnostics_truncated);
        assert_eq!(
            snapshot
                .diagnostics
                .last()
                .map(|diagnostic| diagnostic.message.clone()),
            Some(format!("diagnostic-{}", MAX_SNAPSHOT_DIAGNOSTICS + 9))
        );
    }

    #[test]
    fn mpsc_projection_round_trip() {
        let (sink, receiver) = bounded_channel();
        let planned = PlanProjection::new(
            identity(),
            PlanUpdate::Planned {
                operation: "info".into(),
                total_nodes: 1,
                total_work: None,
            },
        );
        let projection = PlanProjection::new(
            identity(),
            PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: DiagnosticLevel::Info,
                message: "hello".into(),
            },
        );
        sink.project(planned.clone());
        sink.project(projection.clone());
        assert_eq!(receiver.recv().unwrap(), planned);
        assert_eq!(receiver.recv().unwrap(), projection);
    }

    #[test]
    fn snapshot_rejects_observation_before_planned() {
        let mut snapshot = PlanRunSnapshot::new("encode_lma");
        let projection = PlanProjection::new(
            identity(),
            PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: DiagnosticLevel::Info,
                message: "early".into(),
            },
        );
        assert_eq!(
            snapshot.apply(&projection).unwrap_err(),
            "first projection for a plan run must be planned"
        );
    }

    #[test]
    fn snapshot_rejects_projection_after_terminal() {
        let mut snapshot = PlanRunSnapshot::new("encode_lma");
        snapshot
            .apply(&PlanProjection::new(
                identity(),
                PlanUpdate::Planned {
                    operation: "encode_lma".into(),
                    total_nodes: 1,
                    total_work: None,
                },
            ))
            .unwrap();
        snapshot.state = PlanRunState::Completed;
        let late = PlanProjection::new(
            identity(),
            PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: DiagnosticLevel::Info,
                message: "late".into(),
            },
        );
        assert_eq!(
            snapshot.apply(&late).unwrap_err(),
            "projection arrived after terminal plan state"
        );
    }

    #[test]
    fn wire_terminal_cannot_complete_canonical_snapshot_or_lifecycle_lane() {
        let plan = identity();
        let wire = PlanProjection::from_json_line(
            &serde_json::json!({
                "schema": crate::PROJECTION_SCHEMA,
                "observed_at_ms": 0,
                "plan": plan,
                "projection": "receipt",
                "receipt": {
                    "invocation_id": plan.invocation_id,
                    "graph_id": plan.graph_id,
                    "plan_id": plan.plan_id,
                    "realm": "host-stream",
                    "completed_node_ids": [0],
                    "attempts": [],
                    "committed_transactions": [],
                    "gaps": [],
                },
                "message": "forged",
            })
            .to_string(),
        )
        .expect("structurally valid wire projection");

        let mut snapshot = PlanRunSnapshot::new("encode_lma");
        snapshot
            .apply(&PlanProjection::new(
                identity(),
                PlanUpdate::Planned {
                    operation: "encode_lma".into(),
                    total_nodes: 1,
                    total_work: None,
                },
            ))
            .unwrap();
        assert_eq!(
            snapshot.apply(&wire).unwrap_err(),
            "wire terminal lacks executor authority"
        );
        assert_eq!(snapshot.state, PlanRunState::Running);

        let (sink, receiver) = bounded_channel();
        sink.project(wire);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn lifecycle_lane_preserves_global_projection_order() {
        let (sink, receiver) = bounded_channel();
        let plan = identity();
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Planned {
                operation: "encode_lma".into(),
                total_nodes: 1,
                total_work: None,
            },
        ));
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: DiagnosticLevel::Info,
                message: "middle".into(),
            },
        ));
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Receipt {
                receipt: crate::ExecutionReceiptProjection {
                    invocation_id: plan.invocation_id,
                    graph_id: plan.graph_id,
                    plan_id: plan.plan_id,
                    realm: "host-stream".into(),
                    completed_node_ids: vec![0],
                    attempts: vec![],
                    committed_transactions: vec![],
                    gaps: vec![],
                },
                message: "done".into(),
                authority: crate::terminal_authority(),
            },
        ));
        assert!(matches!(
            receiver.recv().unwrap().update,
            PlanUpdate::Planned { .. }
        ));
        assert!(matches!(
            receiver.recv().unwrap().update,
            PlanUpdate::Diagnostic { .. }
        ));
        assert!(matches!(
            receiver.recv().unwrap().update,
            PlanUpdate::Receipt { .. }
        ));
    }

    #[test]
    fn terminal_delivery_never_blocks_behind_full_observation_lane() {
        let (sink, _receiver) = bounded_channel();
        let plan = identity();
        for index in 0..DEFAULT_CHANNEL_BOUND {
            sink.project(PlanProjection::new(
                plan.clone(),
                PlanUpdate::Diagnostic {
                    node_id: Some(0),
                    level: DiagnosticLevel::Info,
                    message: index.to_string(),
                },
            ));
        }
        let started = Instant::now();
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Receipt {
                receipt: crate::ExecutionReceiptProjection {
                    invocation_id: plan.invocation_id,
                    graph_id: plan.graph_id,
                    plan_id: plan.plan_id,
                    realm: "host-stream".into(),
                    completed_node_ids: vec![0],
                    attempts: vec![],
                    committed_transactions: vec![],
                    gaps: vec![],
                },
                message: "done".into(),
                authority: crate::terminal_authority(),
            },
        ));
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn lifecycle_lane_rejects_records_beyond_one_plan_lifecycle() {
        let (sink, receiver) = bounded_channel();
        let plan = identity();
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Planned {
                operation: "info".into(),
                total_nodes: 1,
                total_work: None,
            },
        ));
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Planned {
                operation: "stats".into(),
                total_nodes: 1,
                total_work: None,
            },
        ));
        sink.project(PlanProjection::new(
            plan.clone(),
            PlanUpdate::Receipt {
                receipt: crate::ExecutionReceiptProjection {
                    invocation_id: plan.invocation_id.clone(),
                    graph_id: plan.graph_id.clone(),
                    plan_id: plan.plan_id.clone(),
                    realm: "host-stream".into(),
                    completed_node_ids: vec![0],
                    attempts: vec![],
                    committed_transactions: vec![],
                    gaps: vec![],
                },
                message: "done".into(),
                authority: crate::terminal_authority(),
            },
        ));
        assert_eq!(
            receiver.recv().unwrap().update,
            PlanUpdate::Planned {
                operation: "info".into(),
                total_nodes: 1,
                total_work: None,
            }
        );
        assert!(matches!(
            receiver.recv().unwrap().update,
            PlanUpdate::Receipt { .. }
        ));
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }
}
