use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HASH_DOMAIN: &[u8] = b"org.quitetall.lamquant.lml-optimum-v2.peer-source-v1\0";

fn collect_files(root: &Path, output: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", root.display()))
        .map(|entry| entry.expect("source directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_files(&path, output);
        } else if path.is_file() {
            output.push(path);
        }
    }
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn digest_hex(material: &[u8]) -> String {
    Sha256::digest(material)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn emit(name: &str, value: &str) {
    assert!(!value.contains(['\n', '\r']), "invalid build identity");
    println!("cargo:rustc-env={name}={value}");
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let mut files = vec![manifest.join("Cargo.toml"), manifest.join("build.rs")];
    collect_files(&manifest.join("src"), &mut files);
    files.sort();

    let mut source = Sha256::new();
    source.update(HASH_DOMAIN);
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&manifest).unwrap_or(&path);
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        source.update((relative.len() as u64).to_le_bytes());
        source.update(relative.as_bytes());
        source.update((bytes.len() as u64).to_le_bytes());
        source.update(bytes);
    }
    let source_id = source
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_version = command_output(&rustc, &["--version", "--verbose"])
        .lines()
        .collect::<Vec<_>>()
        .join(";");
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned());
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_owned());
    let debug = env::var("DEBUG").unwrap_or_else(|_| "unknown".to_owned());
    let panic_strategy = env::var("CARGO_CFG_PANIC").unwrap_or_else(|_| "unknown".to_owned());
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PARALLEL");
    let feature_set = if env::var_os("CARGO_FEATURE_PARALLEL").is_some() {
        "parallel"
    } else {
        "none"
    };
    let build_id = digest_hex(
        format!(
            "source={source_id};target={target};profile={profile};opt={opt_level};debug={debug};panic={panic_strategy};features={feature_set};rustc={rustc_version};rustflags={rustflags}"
        )
        .as_bytes(),
    );

    emit("LAMQUANT_OPTIMUM_V2_PEER_SOURCE_ID", &source_id);
    emit("LAMQUANT_OPTIMUM_V2_PEER_BUILD_ID", &build_id);
    emit("LAMQUANT_OPTIMUM_V2_PEER_FEATURE_SET", feature_set);
}
