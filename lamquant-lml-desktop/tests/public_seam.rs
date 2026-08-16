//! Structural contract for the MCU/Desktop ownership boundary.

use std::fs;
use std::path::Path;

const ORCHESTRATION_INTERNALS: &[&str] = &[
    "prepare_encode",
    "encode_one_channel",
    "finalize_channels",
    "parse_lml_channels",
    "DecodePlan",
    "synthesize_channel_signal",
    "EncodePrep",
    "EncodeShape",
    "ChannelEncodeOutput",
];

fn read_rust_tree(root: &Path) -> String {
    let mut source = String::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            source.push_str(&read_rust_tree(&path));
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            source.push_str(&fs::read_to_string(path).unwrap());
            source.push('\n');
        }
    }
    source
}

#[test]
fn desktop_does_not_import_codec_orchestration_internals() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let production = read_rust_tree(&root.join("src"));

    for symbol in ORCHESTRATION_INTERNALS {
        assert!(
            !production.contains(symbol),
            "Desktop production source imports codec orchestration internal {symbol}"
        );
    }
    assert!(!production.contains("pub use lamquant_lml_mcu as"));
    assert!(!production.contains("pub use lamquant_lml_mcu::*"));
    assert!(!production.contains("assemble_lml_packet"));
}

#[test]
fn codec_orchestration_helpers_are_crate_private() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lml = fs::read_to_string(root.join("../lamquant-lml-mcu/src/lml.rs")).unwrap();

    for symbol in ORCHESTRATION_INTERNALS {
        let function = format!("pub(crate) fn {symbol}");
        let type_declaration = format!("pub(crate) enum {symbol}");
        let struct_declaration = format!("pub(crate) struct {symbol}");
        assert!(
            lml.contains(&function)
                || lml.contains(&type_declaration)
                || lml.contains(&struct_declaration),
            "codec orchestration internal {symbol} is not crate-private"
        );
        assert!(!lml.contains(&format!("pub fn {symbol}")));
        assert!(!lml.contains(&format!("pub enum {symbol}")));
        assert!(!lml.contains(&format!("pub struct {symbol}")));
        assert!(!lml.contains(&format!("pub use {symbol}")));
    }
    assert!(!lml.contains("pub fn assemble_lml_packet"));
    let mcu_lib = fs::read_to_string(root.join("../lamquant-lml-mcu/src/lib.rs")).unwrap();
    assert!(!mcu_lib.contains("pub mod parallel"));
    assert!(!lml.contains("pub fn compress_into"));
    assert!(!lml.contains("pub fn decompress_from"));
}
