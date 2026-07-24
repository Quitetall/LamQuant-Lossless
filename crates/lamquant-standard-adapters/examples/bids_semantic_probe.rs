// SPDX-License-Identifier: AGPL-3.0-or-later
//! Round trip a BIDS dataset directory and report what the adapter recovered.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::BidsSemanticAdapter;
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(AdapterError::MissingPayload(content_id))
    }
}

fn collect(root: &Path, base: &Path, into: &mut Vec<ForeignEntry>) -> std::io::Result<()> {
    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, into)?;
        } else {
            into.push(ForeignEntry {
                path: path
                    .strip_prefix(root)
                    .expect("walked path is under the root")
                    .to_string_lossy()
                    .into_owned(),
                media_type: None,
                bytes: fs::read(&path)?,
            });
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or("missing input directory")?);
    let output = PathBuf::from(arguments.next().ok_or("missing output directory")?);
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let mut entries = Vec::new();
    collect(&input, &input, &mut entries)?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    let source = ForeignObject {
        profile: ProfileId("bids.1.11.1".to_owned()),
        entries,
    };
    let adapter = BidsSemanticAdapter::new(1 << 30);
    let report = adapter.inspect(&source)?;
    let imported = adapter.import(&source, ValidationLimits::default())?;
    let payloads = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );
    let plan = adapter.plan_export(&imported.dataset)?;
    let (restored, receipt) = adapter.export(&imported.dataset, &plan, &payloads)?;
    if !receipt.exact_source_restoration || !receipt.semantic_equivalence {
        return Err("BIDS adapter did not issue an exact semantic receipt".into());
    }
    for entry in &restored.entries {
        let target = output.join(&entry.path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, &entry.bytes)?;
    }
    println!(
        "{{\"recordings\":{},\"modalities\":{},\"events\":{},\"electrodes\":{},\"derivatives\":{},\"members\":{}}}",
        report.required_resources["recordings"],
        report.required_resources["modalities"],
        report.required_resources["events"],
        report.required_resources["electrodes"],
        report.required_resources["derivatives"],
        restored.entries.len(),
    );
    Ok(())
}
