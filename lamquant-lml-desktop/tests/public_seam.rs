//! Structural contract for the MCU/Desktop ownership boundary.

use std::fs;
use std::path::Path;

const INTERNAL_SYMBOLS: &[&str] = &[
    "prepare_encode",
    "encode_one_channel",
    "finalize_channels",
    "assemble_lml_packet",
    "parse_lml_channels",
    "DecodePlan",
    "synthesize_channel_signal",
];

#[test]
fn desktop_does_not_import_codec_orchestration_internals() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let production = [
        "src/backend.rs",
        "src/io.rs",
        "src/lib.rs",
        "src/parallel.rs",
    ]
    .into_iter()
    .map(|path| fs::read_to_string(root.join(path)).unwrap())
    .collect::<Vec<_>>()
    .join("\n");

    for symbol in INTERNAL_SYMBOLS {
        assert!(
            !production.contains(symbol),
            "Desktop production source imports codec orchestration internal {symbol}"
        );
    }
    assert!(!production.contains("pub use lamquant_lml_mcu as"));
    assert!(!production.contains("pub use lamquant_lml_mcu::*"));
}

#[test]
fn codec_orchestration_helpers_are_crate_private() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lml = fs::read_to_string(root.join("../lamquant-lml-mcu/src/lml.rs")).unwrap();

    for symbol in INTERNAL_SYMBOLS {
        let function = format!("pub(crate) fn {symbol}");
        let type_declaration = format!("pub(crate) enum {symbol}");
        assert!(
            lml.contains(&function) || lml.contains(&type_declaration),
            "codec orchestration internal {symbol} is not crate-private"
        );
    }
    assert!(!lml.contains("pub fn compress_into"));
    assert!(!lml.contains("pub fn decompress_from"));
}
