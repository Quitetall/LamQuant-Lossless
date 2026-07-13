//! Structured host workflows shared by CLI, JSON-event, and TUI adapters.

use crate::lma::ArchiveVerification;
use std::error::Error;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type WorkflowError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone)]
pub enum InspectionReport {
    Archive(Vec<crate::lma::ArchiveEntry>),
    Container(crate::container_reader::ContainerInspection),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTarget {
    Lml,
    Lma,
}

#[derive(Debug, Clone)]
pub enum VerificationOutcome {
    Lml {
        channels: usize,
        samples: usize,
    },
    Lma(ArchiveVerification),
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
            VerificationOutcome::Lml { .. } => VerificationTarget::Lml,
            VerificationOutcome::Lma(_) => VerificationTarget::Lma,
            VerificationOutcome::Failed { target, .. } => *target,
        }
    }

    pub fn passed(&self) -> bool {
        match &self.outcome {
            VerificationOutcome::Lml { .. } => true,
            VerificationOutcome::Lma(result) => result.passed(),
            VerificationOutcome::Failed { .. } => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub input: PathBuf,
    pub items: Vec<VerificationItem>,
    legacy_single_archive_dispatch: bool,
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
            .any(|item| item.target() == VerificationTarget::Lma)
    }

    pub fn is_success(&self) -> bool {
        self.failed() == 0
    }

    pub fn archive_item(&self) -> Option<&VerificationItem> {
        self.items
            .iter()
            .find(|item| item.target() == VerificationTarget::Lma)
    }

    /// Preserve the historical one-archive rendering path for LMA1 inputs.
    /// LMA2 already shipped through the batch renderer, so changing it would
    /// alter stable CLI output despite both formats sharing verification logic.
    pub fn uses_legacy_single_archive_rendering(&self) -> bool {
        self.legacy_single_archive_dispatch
    }

    #[cfg(feature = "tui")]
    pub fn op_events(&self) -> Vec<lamquant_ops::OpEvent> {
        let total = self.items.len() as u64;
        let mut events = Vec::with_capacity(self.items.len() * 2);
        for (index, item) in self.items.iter().enumerate() {
            let (samples, n_channels) = match &item.outcome {
                VerificationOutcome::Lml { channels, samples } => {
                    (Some(*samples as u64), Some(*channels as u32))
                }
                _ => (None, None),
            };
            events.push(lamquant_ops::OpEvent::Progress {
                ts_ms: lamquant_ops::OpEvent::now_ms(),
                current: (index + 1) as u64,
                total,
                message: item.path.display().to_string(),
            });
            events.push(lamquant_ops::OpEvent::FileDone {
                ts_ms: lamquant_ops::OpEvent::now_ms(),
                path: item.path.display().to_string(),
                success: item.passed(),
                ms: item.elapsed_ms.round() as u64,
                cr: None,
                bytes_in: None,
                bytes_out: None,
                samples,
                duration_s: None,
                n_channels,
                sample_rate: None,
                sha256: None,
                n_windows: None,
            });
        }
        events
    }
}

/// Verify a file or directory without rendering output or terminating the process.
pub fn verify_path(input: &Path, recursive: bool) -> Result<VerificationReport, WorkflowError> {
    if !input.exists() {
        return Err(format!("input path does not exist: {}", input.display()).into());
    }

    if input.is_file() {
        let format = crate::lma::probe_format(input).map_err(|error| {
            format!(
                "cmd_verify: cannot open {} for magic-byte check: {}",
                input.display(),
                error
            )
        })?;
        if format == Some(crate::lma::LmaFormat::V1) {
            let mut report = verify_archive(input)?;
            report.legacy_single_archive_dispatch = true;
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
        legacy_single_archive_dispatch: false,
    })
}

/// Verify one archive through the same result interface used by mixed batches.
pub fn verify_archive(input: &Path) -> Result<VerificationReport, WorkflowError> {
    let started = Instant::now();
    let result = crate::lma::verify_archive(input)?;
    Ok(VerificationReport {
        input: input.to_path_buf(),
        items: vec![VerificationItem {
            path: input.to_path_buf(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            outcome: VerificationOutcome::Lma(result),
        }],
        legacy_single_archive_dispatch: false,
    })
}

/// Inspect a container without exposing format-specific parsing to adapters.
pub fn inspect_path(input: &Path) -> Result<InspectionReport, WorkflowError> {
    if crate::lma::probe_format(input)?.is_some() {
        let archive = crate::lma::LmaArchive::open(input)?;
        return Ok(InspectionReport::Archive(archive.entries().to_vec()));
    }
    match crate::container_reader::ContainerReader::open(input) {
        Ok(reader) => Ok(InspectionReport::Container(reader.inspection())),
        Err(crate::error::LmlError::InvalidMagic(magic)) => Err(format!(
            "Not LML or LMA (magic: {:?}). Expected leading bytes `LML1` or `BCS1` or `LMA1`.",
            magic
        )
        .into()),
        Err(error) => Err(error.into()),
    }
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
        VerificationTarget::Lma
    } else {
        VerificationTarget::Lml
    };
    let outcome = match target {
        VerificationTarget::Lma => match crate::lma::verify_archive(&path) {
            Ok(result) => VerificationOutcome::Lma(result),
            Err(error) => VerificationOutcome::Failed {
                target,
                error: error.to_string(),
            },
        },
        VerificationTarget::Lml => match crate::container::read_file(&path) {
            Ok((signal, _)) => VerificationOutcome::Lml {
                channels: signal.len(),
                samples: signal.first().map_or(0, Vec::len),
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
