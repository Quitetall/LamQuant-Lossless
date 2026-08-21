//! Stable JSON projection of compiled plan observations and terminal receipts.

use blut_graph_core::{
    ExecutionError, ExecutionFailure, ExecutionReceipt, GapReceipt, StructuredFailure,
};
use serde::{Deserialize, Serialize};

pub const PROJECTION_SCHEMA: &str = "org.quitetall.lamquant.plan-projection/v1";
pub const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanIdentity {
    pub graph_id: String,
    pub plan_id: String,
    pub invocation_id: String,
}

impl PlanIdentity {
    pub fn new(graph_id: &[u8; 32], plan_id: &[u8; 32], invocation_id: &[u8; 32]) -> Self {
        Self {
            graph_id: hex(graph_id),
            plan_id: hex(plan_id),
            invocation_id: hex(invocation_id),
        }
    }

    pub fn is_valid(&self) -> bool {
        [&self.graph_id, &self.plan_id, &self.invocation_id]
            .into_iter()
            .all(|value| is_lower_hex_256(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiagnosticLevel {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactProjection {
    pub path: String,
    pub success: bool,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_in: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_out: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionAttemptProjection {
    pub step_id: u32,
    pub node_ids: Vec<u32>,
    pub kernel_id: u32,
    pub implementation_id: String,
    pub attempts: u32,
    pub kernel_succeeded: bool,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GapProjection {
    pub step_id: u32,
    pub node_ids: Vec<u32>,
    pub output_index: u32,
    pub offset: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<u64>,
    pub domain: String,
    pub code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionReceiptProjection {
    pub invocation_id: String,
    pub graph_id: String,
    pub plan_id: String,
    pub realm: String,
    pub completed_node_ids: Vec<u32>,
    pub attempts: Vec<ExecutionAttemptProjection>,
    pub committed_transactions: Vec<String>,
    pub gaps: Vec<GapProjection>,
}

impl From<&ExecutionReceipt> for ExecutionReceiptProjection {
    fn from(receipt: &ExecutionReceipt) -> Self {
        Self {
            invocation_id: hex(&receipt.invocation_id),
            graph_id: hex(&receipt.graph_id.0),
            plan_id: hex(&receipt.plan_id.0),
            realm: match receipt.realm {
                blut_graph_core::ExecutionRealm::McuAot => "mcu-aot",
                blut_graph_core::ExecutionRealm::HostStream => "host-stream",
                blut_graph_core::ExecutionRealm::BlutDurable => "blut-durable",
            }
            .into(),
            completed_node_ids: receipt.completed_nodes.iter().map(|id| id.0).collect(),
            attempts: receipt
                .attempts
                .iter()
                .map(|attempt| ExecutionAttemptProjection {
                    step_id: attempt.step.0,
                    node_ids: attempt.semantic_nodes.iter().map(|id| id.0).collect(),
                    kernel_id: attempt.kernel.0,
                    implementation_id: hex(&attempt.implementation_id.0),
                    attempts: attempt.attempts,
                    kernel_succeeded: attempt.kernel_succeeded,
                    completed: attempt.completed,
                })
                .collect(),
            committed_transactions: receipt.committed_transactions.clone(),
            gaps: receipt.gaps.iter().map(GapProjection::from).collect(),
        }
    }
}

impl From<&GapReceipt> for GapProjection {
    fn from(receipt: &GapReceipt) -> Self {
        Self {
            step_id: receipt.step.0,
            node_ids: receipt.semantic_nodes.iter().map(|id| id.0).collect(),
            output_index: receipt.gap.output_index,
            offset: receipt.gap.offset,
            length: receipt.gap.length,
            domain: receipt.gap.domain.clone(),
            code: receipt.gap.code.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureProjection {
    pub domain: String,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl FailureProjection {
    pub fn process(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            domain: "org.quitetall.lamquant.process".into(),
            code: code.into(),
            message: message.into(),
            retryable: false,
        }
    }

    fn from_execution(error: &ExecutionError) -> Self {
        match error {
            ExecutionError::KernelFailed { failure, .. } => Self::from(failure),
            other => Self {
                domain: "org.quitetall.blut.graph.execute".into(),
                code: execution_error_code(other).into(),
                message: other.to_string(),
                retryable: false,
            },
        }
    }
}

impl From<&StructuredFailure> for FailureProjection {
    fn from(failure: &StructuredFailure) -> Self {
        Self {
            domain: failure.domain.clone(),
            code: failure.code.clone(),
            message: failure.message.clone(),
            retryable: failure.retryable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "projection", rename_all = "kebab-case")]
pub enum PlanUpdate {
    Planned {
        operation: String,
        total_nodes: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_work: Option<u64>,
    },
    Progress {
        node_id: u32,
        current: u64,
        total: u64,
        message: String,
    },
    Artifact {
        node_id: u32,
        artifact: ArtifactProjection,
    },
    Receipt {
        receipt: ExecutionReceiptProjection,
        message: String,
        #[serde(skip, default = "wire_terminal_authority")]
        authority: TerminalProjectionAuthority,
    },
    Failure {
        receipt: ExecutionReceiptProjection,
        failure: FailureProjection,
        cancelled: bool,
        #[serde(skip, default = "wire_terminal_authority")]
        authority: TerminalProjectionAuthority,
    },
    Diagnostic {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<u32>,
        level: DiagnosticLevel,
        message: String,
    },
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalProjectionAuthority {
    source: TerminalAuthoritySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalAuthoritySource {
    Executor,
    Wire,
}

pub fn terminal_authority() -> TerminalProjectionAuthority {
    TerminalProjectionAuthority {
        source: TerminalAuthoritySource::Executor,
    }
}

fn wire_terminal_authority() -> TerminalProjectionAuthority {
    TerminalProjectionAuthority {
        source: TerminalAuthoritySource::Wire,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanProjection {
    pub schema: String,
    pub observed_at_ms: i64,
    pub plan: PlanIdentity,
    #[serde(flatten)]
    pub update: PlanUpdate,
}

impl PlanProjection {
    pub fn now_ms() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0)
    }

    pub fn new(plan: PlanIdentity, update: PlanUpdate) -> Self {
        Self {
            schema: PROJECTION_SCHEMA.into(),
            observed_at_ms: Self::now_ms(),
            plan,
            update,
        }
    }

    pub fn to_json_line(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| format!("plan projection serialize: {error}"))
    }

    pub fn from_json_line(line: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("plan projection parse: {error}"))?;
        if contains_json_null(&value) {
            return Err("plan projection does not permit JSON null".into());
        }
        validate_top_level_keys(&value)?;
        let projection: Self = serde_json::from_value(value)
            .map_err(|error| format!("plan projection parse: {error}"))?;
        projection.validate()?;
        Ok(projection)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECTION_SCHEMA {
            return Err(format!(
                "unsupported plan projection schema: {}",
                self.schema
            ));
        }
        if self.observed_at_ms < 0 || self.observed_at_ms as u64 > MAX_SAFE_JSON_INTEGER {
            return Err("plan projection timestamp must be a nonnegative safe integer".into());
        }
        if !self.plan.is_valid() {
            return Err("invalid plan projection identity".into());
        }
        match &self.update {
            PlanUpdate::Planned { operation, .. }
                if !crate::operation_id::is_canonical_operation_id(operation) =>
            {
                Err(format!(
                    "planned projection operation is not registered: {operation}"
                ))
            }
            PlanUpdate::Planned {
                total_nodes,
                total_work,
                ..
            } if *total_nodes == 0
                || total_work.is_some_and(|value| value > MAX_SAFE_JSON_INTEGER) =>
            {
                Err("invalid planned projection".into())
            }
            PlanUpdate::Progress { total, current, .. }
                if *total == 0
                    || current > total
                    || *current > MAX_SAFE_JSON_INTEGER
                    || *total > MAX_SAFE_JSON_INTEGER =>
            {
                Err("invalid progress projection".into())
            }
            PlanUpdate::Artifact { artifact, .. } => validate_artifact(artifact),
            PlanUpdate::Receipt { receipt, .. } => self.validate_receipt(receipt),
            PlanUpdate::Failure {
                receipt, failure, ..
            } => {
                self.validate_receipt(receipt)?;
                if failure.domain.is_empty() || failure.code.is_empty() {
                    return Err("failure projection requires domain and code".into());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.update,
            PlanUpdate::Receipt { .. } | PlanUpdate::Failure { .. }
        )
    }

    pub fn terminal_is_executor_issued(&self) -> bool {
        match &self.update {
            PlanUpdate::Receipt { authority, .. } | PlanUpdate::Failure { authority, .. } => {
                authority.source == TerminalAuthoritySource::Executor
            }
            _ => false,
        }
    }

    fn validate_receipt(&self, receipt: &ExecutionReceiptProjection) -> Result<(), String> {
        if receipt.graph_id != self.plan.graph_id
            || receipt.plan_id != self.plan.plan_id
            || receipt.invocation_id != self.plan.invocation_id
        {
            return Err("terminal receipt identity does not match projection plan".into());
        }
        if !matches!(
            receipt.realm.as_str(),
            "mcu-aot" | "host-stream" | "blut-durable"
        ) {
            return Err("terminal receipt has unknown execution realm".into());
        }
        if receipt
            .attempts
            .iter()
            .any(|attempt| !is_lower_hex_256(&attempt.implementation_id))
        {
            return Err("terminal receipt has invalid implementation identity".into());
        }
        if receipt.gaps.iter().any(|gap| {
            gap.offset > MAX_SAFE_JSON_INTEGER
                || gap
                    .length
                    .is_some_and(|value| value == 0 || value > MAX_SAFE_JSON_INTEGER)
                || gap.domain.is_empty()
                || gap.code.is_empty()
        }) {
            return Err("terminal receipt has invalid gap evidence".into());
        }
        Ok(())
    }

    pub fn completed(receipt: &ExecutionReceipt, message: impl Into<String>) -> Self {
        let plan = PlanIdentity::new(
            &receipt.graph_id.0,
            &receipt.plan_id.0,
            &receipt.invocation_id,
        );
        Self::new(
            plan,
            PlanUpdate::Receipt {
                receipt: receipt.into(),
                message: message.into(),
                authority: terminal_authority(),
            },
        )
    }

    pub fn failed(failure: &ExecutionFailure, cancelled: bool) -> Self {
        let plan = PlanIdentity::new(
            &failure.receipt.graph_id.0,
            &failure.receipt.plan_id.0,
            &failure.receipt.invocation_id,
        );
        Self::new(
            plan,
            PlanUpdate::Failure {
                receipt: (&failure.receipt).into(),
                failure: FailureProjection::from_execution(&failure.error),
                cancelled,
                authority: terminal_authority(),
            },
        )
    }
}

fn validate_top_level_keys(value: &serde_json::Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "plan projection must be a JSON object".to_string())?;
    let projection = object
        .get("projection")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "plan projection discriminator must be a string".to_string())?;
    let variant_keys: &[&str] = match projection {
        "planned" => &["operation", "total_nodes", "total_work"],
        "progress" => &["node_id", "current", "total", "message"],
        "artifact" => &["node_id", "artifact"],
        "receipt" => &["receipt", "message"],
        "failure" => &["receipt", "failure", "cancelled"],
        "diagnostic" => &["node_id", "level", "message"],
        _ => return Err(format!("unknown plan projection variant: {projection}")),
    };
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "schema" | "observed_at_ms" | "plan" | "projection"
        ) && !variant_keys.contains(&key.as_str())
        {
            return Err(format!("unknown field in {projection} projection: {key}"));
        }
    }
    Ok(())
}

fn contains_json_null(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Array(items) => items.iter().any(contains_json_null),
        serde_json::Value::Object(fields) => fields.values().any(contains_json_null),
        _ => false,
    }
}

fn execution_error_code(error: &ExecutionError) -> &'static str {
    match error {
        ExecutionError::UnknownKernel(_) => "unknown-kernel",
        ExecutionError::MissingBuffer(_) => "missing-buffer",
        ExecutionError::MissingInvocation(_) => "missing-invocation",
        ExecutionError::UnexpectedInvocation(_) => "unexpected-invocation",
        ExecutionError::OutputArity { .. } => "output-arity",
        ExecutionError::KernelFailed { .. } => "kernel-failed",
        ExecutionError::UndeclaredFailure(_, _) => "undeclared-failure",
        ExecutionError::UnsafeRetry(_) => "unsafe-retry",
        ExecutionError::TransactionPrepare(_) => "transaction-prepare",
        ExecutionError::TransactionCommit(_) => "transaction-commit",
        ExecutionError::InvalidGap(_) => "invalid-gap",
        ExecutionError::StatefulPlanUnsupported => "stateful-plan-unsupported",
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn is_lower_hex_256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_artifact(artifact: &ArtifactProjection) -> Result<(), String> {
    if artifact.bytes_in.is_some() != artifact.bytes_out.is_some() {
        return Err("artifact byte telemetry must be complete or absent".into());
    }
    if artifact.elapsed_ms > MAX_SAFE_JSON_INTEGER
        || artifact
            .bytes_in
            .is_some_and(|value| value > MAX_SAFE_JSON_INTEGER)
        || artifact
            .bytes_out
            .is_some_and(|value| value > MAX_SAFE_JSON_INTEGER)
        || artifact
            .samples
            .is_some_and(|value| value > MAX_SAFE_JSON_INTEGER)
    {
        return Err("artifact integer telemetry exceeds safe JSON range".into());
    }
    if artifact
        .compression_ratio
        .is_some_and(|value| !value.is_finite() || value < 0.0)
        || artifact
            .duration_seconds
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || artifact
            .sample_rate_hz
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err("artifact telemetry must contain finite nonnegative numbers".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut_graph_core::{
        ExecutionAttempt, ExecutionRealm, GraphId, ImplementationId, KernelId, NodeId, PlanId,
        StepId,
    };

    fn receipt() -> ExecutionReceipt {
        ExecutionReceipt {
            invocation_id: [0x11; 32],
            graph_id: GraphId([0x22; 32]),
            plan_id: PlanId([0x33; 32]),
            realm: ExecutionRealm::HostStream,
            completed_nodes: vec![NodeId(7)],
            attempts: vec![ExecutionAttempt {
                step: StepId(0),
                semantic_nodes: vec![NodeId(7)],
                kernel: KernelId(9),
                implementation_id: ImplementationId([0x44; 32]),
                attempts: 1,
                kernel_succeeded: true,
                completed: true,
            }],
            committed_transactions: vec![],
            gaps: vec![],
        }
    }

    #[test]
    fn terminal_projection_round_trips_and_binds_receipt_identity() {
        let projection = PlanProjection::completed(&receipt(), "done");
        let line = projection.to_json_line().unwrap();
        let decoded = PlanProjection::from_json_line(&line).unwrap();
        assert!(projection.terminal_is_executor_issued());
        assert!(!decoded.terminal_is_executor_issued());
        assert_eq!(decoded.to_json_line().unwrap(), line);
        assert!(line.contains(PROJECTION_SCHEMA));
        assert!(line.contains("\"completed_node_ids\":[7]"));
    }

    #[test]
    fn mismatched_terminal_identity_fails_closed() {
        let mut projection = PlanProjection::completed(&receipt(), "done");
        projection.plan.plan_id = "00".repeat(32);
        assert_eq!(
            projection.validate().unwrap_err(),
            "terminal receipt identity does not match projection plan"
        );
    }

    #[test]
    fn incomplete_artifact_sizes_fail_closed() {
        let projection = PlanProjection::new(
            PlanIdentity::new(&[1; 32], &[2; 32], &[3; 32]),
            PlanUpdate::Artifact {
                node_id: 0,
                artifact: ArtifactProjection {
                    path: "a.lml".into(),
                    success: true,
                    elapsed_ms: 1,
                    compression_ratio: None,
                    bytes_in: Some(4),
                    bytes_out: None,
                    samples: None,
                    duration_seconds: None,
                    channel_count: None,
                    sample_rate_hz: None,
                    sha256: None,
                    window_count: None,
                },
            },
        );
        assert!(projection.validate().is_err());
    }

    #[test]
    fn unsafe_json_integer_fails_closed() {
        let projection = PlanProjection::new(
            PlanIdentity::new(&[1; 32], &[2; 32], &[3; 32]),
            PlanUpdate::Progress {
                node_id: 0,
                current: MAX_SAFE_JSON_INTEGER + 1,
                total: MAX_SAFE_JSON_INTEGER + 1,
                message: "unsafe".into(),
            },
        );
        assert!(projection.validate().is_err());
        assert!(projection.to_json_line().is_err());
    }

    #[test]
    fn unknown_top_level_and_nested_fields_fail_closed() {
        let plan = PlanIdentity::new(&[1; 32], &[2; 32], &[3; 32]);
        let line = PlanProjection::new(
            plan,
            PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: DiagnosticLevel::Info,
                message: "x".into(),
            },
        )
        .to_json_line()
        .unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&line).unwrap();
        value["surprise"] = serde_json::json!(true);
        assert!(PlanProjection::from_json_line(&value.to_string()).is_err());

        value.as_object_mut().unwrap().remove("surprise");
        value["plan"]["surprise"] = serde_json::json!(true);
        assert!(PlanProjection::from_json_line(&value.to_string()).is_err());
    }

    #[test]
    fn uppercase_identity_fails_schema_parity() {
        let mut projection = PlanProjection::completed(&receipt(), "done");
        projection.plan.graph_id = "AA".repeat(32);
        assert_eq!(
            projection.validate().unwrap_err(),
            "invalid plan projection identity"
        );
    }

    #[test]
    fn planned_projection_rejects_noncanonical_operation_id() {
        let projection = PlanProjection::new(
            PlanIdentity::new(&[1; 32], &[2; 32], &[3; 32]),
            PlanUpdate::Planned {
                operation: "encode".into(),
                total_nodes: 1,
                total_work: None,
            },
        );
        assert_eq!(
            projection.validate().unwrap_err(),
            "planned projection operation is not registered: encode"
        );
    }
}
