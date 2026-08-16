//! Structured host workflows shared by CLI, JSON projections, and TUI adapters.

use crate::{container, lma};
use std::error::Error;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type WorkflowError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone)]
pub enum InspectionReport {
    Archive {
        entries: Vec<lma::ArchiveEntry>,
    },
    Container {
        header: container::ContainerHeader,
        file_size: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTarget {
    Container,
    Archive,
}

#[derive(Debug, Clone)]
pub enum VerificationOutcome {
    Container {
        header: container::ContainerHeader,
        file_size: u64,
    },
    Archive(lma::ArchiveVerification),
    Failed {
        target: VerificationTarget,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub struct VerificationItem {
    pub path: PathBuf,
    pub elapsed_ms: f64,
    pub outcome: VerificationOutcome,
}

impl VerificationItem {
    pub fn target(&self) -> VerificationTarget {
        match &self.outcome {
            VerificationOutcome::Container { .. } => VerificationTarget::Container,
            VerificationOutcome::Archive(_) => VerificationTarget::Archive,
            VerificationOutcome::Failed { target, .. } => *target,
        }
    }

    pub fn passed(&self) -> bool {
        match &self.outcome {
            VerificationOutcome::Container { .. } => true,
            VerificationOutcome::Archive(verification) => verification.passed(),
            VerificationOutcome::Failed { .. } => false,
        }
    }

    #[cfg(feature = "tui")]
    pub fn to_artifact_projection(&self) -> lamquant_ops::ArtifactProjection {
        match &self.outcome {
            VerificationOutcome::Container { header, file_size } => {
                let raw_bytes = header
                    .n_channels
                    .saturating_mul(header.total_samples)
                    .saturating_mul(8);
                let ratio = if *file_size > 0 {
                    raw_bytes as f64 / *file_size as f64
                } else {
                    0.0
                };
                lamquant_ops::ArtifactProjection {
                    path: self.path.display().to_string(),
                    success: true,
                    elapsed_ms: self.elapsed_ms as u64,
                    compression_ratio: Some(ratio),
                    bytes_in: Some(raw_bytes as u64),
                    bytes_out: Some(*file_size),
                    samples: Some(
                        (header.n_channels as u64).saturating_mul(header.total_samples as u64),
                    ),
                    duration_seconds: Some(header.total_samples as f64 / header.sample_rate_hz),
                    channel_count: Some(header.n_channels as u32),
                    sample_rate_hz: Some(header.sample_rate_hz as f32),
                    sha256: None,
                    window_count: Some(header.n_windows as u32),
                }
            }
            VerificationOutcome::Archive(verification) => {
                let reconstructed = verification.reconstructed_bytes();
                lamquant_ops::ArtifactProjection {
                    path: self.path.display().to_string(),
                    success: verification.passed(),
                    elapsed_ms: self.elapsed_ms as u64,
                    compression_ratio: if verification.archive_size > 0 {
                        Some(reconstructed as f64 / verification.archive_size as f64)
                    } else {
                        None
                    },
                    bytes_in: Some(reconstructed),
                    bytes_out: Some(verification.archive_size),
                    samples: None,
                    duration_seconds: None,
                    channel_count: None,
                    sample_rate_hz: None,
                    sha256: None,
                    window_count: None,
                }
            }
            VerificationOutcome::Failed { .. } => lamquant_ops::ArtifactProjection {
                path: self.path.display().to_string(),
                success: false,
                elapsed_ms: self.elapsed_ms as u64,
                compression_ratio: None,
                bytes_in: None,
                bytes_out: None,
                samples: None,
                duration_seconds: None,
                channel_count: None,
                sample_rate_hz: None,
                sha256: None,
                window_count: None,
            },
        }
    }

    #[cfg(feature = "tui")]
    pub fn to_plan_update(&self) -> lamquant_ops::PlanUpdate {
        lamquant_ops::PlanUpdate::Artifact {
            node_id: 0,
            artifact: self.to_artifact_projection(),
        }
    }

    #[cfg(feature = "tui")]
    pub fn to_plan_updates(&self) -> Vec<lamquant_ops::PlanUpdate> {
        let mut updates = vec![self.to_plan_update()];
        if let Some(message) = self.failure_diagnostic() {
            updates.push(lamquant_ops::PlanUpdate::Diagnostic {
                node_id: Some(0),
                level: lamquant_ops::DiagnosticLevel::Error,
                message,
            });
        }
        updates
    }

    #[cfg(feature = "tui")]
    pub fn to_plan_projection(
        &self,
        identity: &lamquant_ops::PlanIdentity,
    ) -> lamquant_ops::PlanProjection {
        lamquant_ops::PlanProjection::new(identity.clone(), self.to_plan_update())
    }

    fn failure_diagnostic(&self) -> Option<String> {
        match &self.outcome {
            VerificationOutcome::Failed { error, .. } => Some(format!(
                "verification failed for {}: {error}",
                self.path.display()
            )),
            VerificationOutcome::Archive(verification) if !verification.passed() => {
                let mut reasons = Vec::new();
                if !verification.archive_hash_matches {
                    reasons.push(format!(
                        "archive SHA-256 mismatch (stored {}, computed {})",
                        verification.archive_sha256, verification.computed_archive_sha256
                    ));
                }
                let failed_entries: Vec<_> = verification
                    .entries
                    .iter()
                    .filter(|entry| !entry.passed)
                    .collect();
                for entry in failed_entries.iter().take(10) {
                    reasons.push(format!("{}: {}", entry.path, entry.detail));
                }
                if failed_entries.len() > 10 {
                    reasons.push(format!(
                        "{} additional entry failures",
                        failed_entries.len() - 10
                    ));
                }
                Some(format!(
                    "verification failed for {}: {}",
                    self.path.display(),
                    reasons.join("; ")
                ))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub input: PathBuf,
    pub items: Vec<VerificationItem>,
    legacy_single_archive_rendering: bool,
}

impl VerificationReport {
    pub fn passed(&self) -> usize {
        self.items.iter().filter(|item| item.passed()).count()
    }

    pub fn failed(&self) -> usize {
        self.items.len() - self.passed()
    }

    pub fn has_archives(&self) -> bool {
        self.items
            .iter()
            .any(|item| item.target() == VerificationTarget::Archive)
    }

    pub fn is_success(&self) -> bool {
        self.failed() == 0
    }

    /// LMA1 historically dispatched directly to archive rendering. LMA2 kept
    /// normal batch output even for one file; preserve both serialized CLIs.
    pub fn uses_legacy_single_archive_rendering(&self) -> bool {
        self.legacy_single_archive_rendering
    }

    #[cfg(feature = "tui")]
    pub fn to_plan_projections(
        &self,
        identity: &lamquant_ops::PlanIdentity,
    ) -> Vec<lamquant_ops::PlanProjection> {
        self.items
            .iter()
            .flat_map(|item| {
                item.to_plan_updates()
                    .into_iter()
                    .map(|update| lamquant_ops::PlanProjection::new(identity.clone(), update))
            })
            .collect()
    }
}

/// Verify file or directory input without side effects.
pub fn verify_path(input: &Path, recursive: bool) -> Result<VerificationReport, WorkflowError> {
    if !input.exists() {
        return Err(format!("input path does not exist: {}", input.display()).into());
    }

    if input.is_file() {
        let format = lma::probe_format(input).map_err(|error| {
            format!(
                "cmd_verify: cannot open {} for magic-byte check: {}",
                input.display(),
                error
            )
        })?;
        if let Some(format) = format {
            let mut report = verify_archive(input)?;
            report.legacy_single_archive_rendering = format == lma::LmaFormat::V1;
            return Ok(report);
        }
    }

    let files = discover_verification_files(input, recursive)?;
    if files.is_empty() {
        return Err(format!(
            "no .lml files found at {} — verify would silently report 0/0 success otherwise",
            input.display()
        )
        .into());
    }

    let items = files.into_iter().map(verify_item).collect();
    Ok(VerificationReport {
        input: input.to_path_buf(),
        items,
        legacy_single_archive_rendering: false,
    })
}

/// Verify one archive through the same result interface used by mixed batches.
pub fn verify_archive(input: &Path) -> Result<VerificationReport, WorkflowError> {
    Ok(VerificationReport {
        input: input.to_path_buf(),
        items: vec![verify_archive_item(input.to_path_buf())],
        legacy_single_archive_rendering: false,
    })
}

/// Inspect one file without exposing format-specific parsing details.
pub fn inspect_path(input: &Path) -> Result<InspectionReport, WorkflowError> {
    if lma::probe_format(input)?.is_some() {
        let archive = lma::LmaArchive::open(input)?;
        return Ok(InspectionReport::Archive {
            entries: archive.entries().to_vec(),
        });
    }

    let data = std::fs::read(input)?;
    let header = container::parse_header(&data)?;
    Ok(InspectionReport::Container {
        header,
        file_size: data.len() as u64,
    })
}

fn discover_verification_files(
    input: &Path,
    recursive: bool,
) -> Result<Vec<PathBuf>, WorkflowError> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }

    let walker = if recursive {
        walkdir::WalkDir::new(input)
    } else {
        walkdir::WalkDir::new(input).max_depth(1)
    };

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry.map_err(|error| {
            format!("verify walk failed beneath {}: {}", input.display(), error)
        })?;
        let matches = {
            let extension = entry.path().extension();
            extension.is_some_and(|value| value.eq_ignore_ascii_case(OsStr::new("lml")))
                || extension.is_some_and(|value| value.eq_ignore_ascii_case(OsStr::new("lma")))
        };
        if matches {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

fn verify_item(path: PathBuf) -> VerificationItem {
    let started = Instant::now();
    let target = if path
        .extension()
        .is_some_and(|value| value.eq_ignore_ascii_case(OsStr::new("lma")))
    {
        VerificationTarget::Archive
    } else {
        VerificationTarget::Container
    };

    let outcome = match target {
        VerificationTarget::Archive => return verify_archive_item(path),
        VerificationTarget::Container => match std::fs::read(&path) {
            Ok(bytes) => match container::parse_header(&bytes) {
                Ok(header) => VerificationOutcome::Container {
                    header,
                    file_size: bytes.len() as u64,
                },
                Err(error) => VerificationOutcome::Failed {
                    target,
                    error: error.to_string(),
                },
            },
            Err(error) => VerificationOutcome::Failed {
                target,
                error: error.to_string(),
            },
        },
    };

    VerificationItem {
        path,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        outcome,
    }
}

fn verify_archive_item(path: PathBuf) -> VerificationItem {
    let started = Instant::now();
    let outcome = match lma::verify_archive(&path) {
        Ok(result) => VerificationOutcome::Archive(result),
        Err(error) => VerificationOutcome::Failed {
            target: VerificationTarget::Archive,
            error: error.to_string(),
        },
    };
    VerificationItem {
        path,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        outcome,
    }
}
