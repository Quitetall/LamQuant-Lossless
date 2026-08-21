// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable JSON projection of compiled plan observations, and the operation-id registry.
//!
//! WHY THIS LIVES IN ABIR. It arrived here from the private `lamquant-ops`
//! crate in the LamQuant meta-repository, and the move is the whole point:
//! `codec-lossless` is PUBLIC and declared three production dependencies on
//! that private repository, so nobody outside the owning account could build
//! it. Measured 2026-08-20, that was one of four such edges across three public
//! repositories (ADR 0185).
//!
//! ABIR is the right home rather than a convenient one. It is public, depends
//! on nothing else in the fleet, and is ALREADY pinned by both codec-lossless
//! and the meta -- so this adds no repository edge and creates no mutual pin,
//! which putting it in codec-lossless would have done. It is also where the
//! wire contracts belong: `PROJECTION_SCHEMA` is a versioned wire identity
//! (`org.quitetall.lamquant.plan-projection/v1`), not application logic.
//!
//! WHAT DELIBERATELY DID NOT COME. The launcher command table, process runner,
//! channel sink and SSH transport stay in `lamquant-ops`. Those are application
//! layer -- they spawn processes and open sockets. What moved is the vocabulary
//! a front-end needs to *describe* an operation, which is pure data: three
//! modules whose only imports are `blut_graph_core` and `serde`.

#![forbid(unsafe_code)]

pub mod op_spec;
pub mod operation_id;
pub mod projection;

pub use op_spec::{op_spec, OpSpec, CODEC_OPERATION_IDS};
pub use operation_id::{
    canonical_operation_ids, install_operation_id, is_canonical_operation_id, BLUT_OPERATION_IDS,
    EXTERNAL_OPERATION_IDS, INSTALL_OPERATION_IDS,
};
pub use projection::{
    terminal_authority, ArtifactProjection, DiagnosticLevel, ExecutionAttemptProjection,
    ExecutionReceiptProjection, FailureProjection, GapProjection, PlanIdentity, PlanProjection,
    PlanUpdate, TerminalProjectionAuthority, MAX_SAFE_JSON_INTEGER, PROJECTION_SCHEMA,
};
