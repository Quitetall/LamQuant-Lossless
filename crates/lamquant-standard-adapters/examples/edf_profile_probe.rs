use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::EdfAdapter;
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
    let bytes = fs::read(&input)?;
    let media_type = if bytes.first() == Some(&0xff) {
        "application/bdf"
    } else {
        "application/edf"
    };
    let source = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: input
                .file_name()
                .ok_or("input has no filename")?
                .to_string_lossy()
                .into_owned(),
            media_type: Some(media_type.to_owned()),
            bytes,
        }],
    };
    let adapter = EdfAdapter::new(1 << 30);
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
        return Err("EDF adapter did not issue an exact semantic receipt".into());
    }
    fs::write(output, &restored.entries[0].bytes)?;
    Ok(())
}
