// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable JSON projection of compiled plan observations, and the operation-id registry.
//!
//! WHY THIS IS PUBLIC. It arrived from the private `lamquant-ops` crate in the
//! LamQuant meta-repository, and the move is the whole point: `codec-lossless`
//! is PUBLIC and declared three production dependencies on that private
//! repository, so nobody outside the owning account could build it. Measured
//! 2026-08-20, that was one of four such edges across three public repositories
//! (ADR 0185). It is also where the wire contracts belong: `PROJECTION_SCHEMA`
//! is a versioned wire identity (`org.quitetall.lamquant.plan-projection/v1`),
//! not application logic.
//!
//! HEADER CORRECTED 2026-08-22. This block previously opened "WHY THIS LIVES IN
//! ABIR" and argued ABIR was the right home because putting the crate in
//! codec-lossless "would have created a mutual pin". The crate is not in ABIR
//! and never was -- `git ls-files` in codec-lossless tracks
//! `crates/lamquant-plan/`. The doc described a plan that changed during
//! execution and was never re-read. The mutual-pin worry did not survive
//! either, for the reason `lamquant-ops/Cargo.toml` gives at its path
//! dependency: the reverse direction (lossless -> meta) is being deleted by
//! this same campaign, so no mutual pin can form.
//!
//! WHAT DELIBERATELY DID NOT COME. The launcher command table, process runner
//! and SSH transport stay in `lamquant-ops`. Those are application layer --
//! they spawn processes and open sockets. What is here is the vocabulary a
//! front-end needs to *describe* an operation and to *receive* observations of
//! one, which is pure data.
//!
//! `sink` JOINED THEM 2026-08-22, and the original list said it would not.
//! That line was written from the module's NAME rather than its contents.
//! Measured, `sink.rs` imports only `std::sync::{mpsc, Mutex}`,
//! `std::time::{Duration, Instant}`, `serde`, and this crate's own projection
//! types. It spawns no process and opens no socket -- it is a bounded mpsc
//! queue plus a snapshot reducer. Keeping it private meant a public front-end
//! could NAME a `PlanProjection` but had no public way to RECEIVE one, which is
//! not a boundary, just an omission.

#![forbid(unsafe_code)]

pub mod op_spec;
pub mod operation_id;
pub mod projection;
pub mod sink;

pub use op_spec::{op_spec, OpSpec, CODEC_OPERATION_IDS};
// Same names, same order as `lamquant-ops` exported them before the move, so a
// consumer that switches `lamquant_ops::` to `lamquant_plan::` needs no other
// edit and gets the identical type.
pub use operation_id::{
    canonical_operation_ids, install_operation_id, is_canonical_operation_id, BLUT_OPERATION_IDS,
    EXTERNAL_OPERATION_IDS, INSTALL_OPERATION_IDS,
};
pub use projection::{
    terminal_authority, ArtifactProjection, DiagnosticLevel, ExecutionAttemptProjection,
    ExecutionReceiptProjection, FailureProjection, GapProjection, PlanIdentity, PlanProjection,
    PlanUpdate, TerminalProjectionAuthority, MAX_SAFE_JSON_INTEGER, PROJECTION_SCHEMA,
};
pub use sink::{
    bounded_channel, CapturedDiagnostic, MpscProjectionSink, PlanProjectionSink, PlanRunSnapshot,
    PlanRunState, ProjectionReceiver,
};
