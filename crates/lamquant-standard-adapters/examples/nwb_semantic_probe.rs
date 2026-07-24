// SPDX-License-Identifier: AGPL-3.0-or-later
//! Round trip an NWB file through the first-class adapter and report what it
//! recovered, so an evidence producer can measure rather than assert.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::NwbAdapter;
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(AdapterError::MissingPayload(content_id))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or("missing input path")?);
    let output = PathBuf::from(arguments.next().ok_or("missing output path")?);
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let source = ForeignObject {
        profile: ProfileId("nwb.2.10.0".to_owned()),
        entries: vec![ForeignEntry {
            path: input
                .file_name()
                .ok_or("input has no filename")?
                .to_string_lossy()
                .into_owned(),
            media_type: Some("application/x-nwb".to_owned()),
            bytes: fs::read(&input)?,
        }],
    };
    let adapter = NwbAdapter::new(1 << 30);
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
        return Err("NWB adapter did not issue an exact semantic receipt".into());
    }
    fs::write(output, &restored.entries[0].bytes)?;
    println!(
        "{{\"series\":{},\"electrodes\":{},\"intervals\":{},\"external_assets\":{},\"streams\":{},\"events\":{},\"clocks\":{}}}",
        report.required_resources["series"],
        report.required_resources["electrodes"],
        report.required_resources["intervals"],
        report.required_resources["external-assets"],
        imported.dataset.streams().len(),
        imported.dataset.events().len(),
        imported.dataset.clocks().len(),
    );
    Ok(())
}
