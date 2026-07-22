use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HASH_DOMAIN: &[u8] = b"org.quitetall.lamquant.abir-codec.source-v1\0";

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

fn emit(name: &str, value: &str) {
    assert!(!value.contains(['\n', '\r']), "invalid build identity");
    println!("cargo:rustc-env={name}={value}");
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let repository = manifest
        .parent()
        .and_then(Path::parent)
        .expect("integration crate must live under codec-lossless/crates");
    let mut files = vec![manifest.join("Cargo.toml"), manifest.join("build.rs")];
    collect_files(&manifest.join("src"), &mut files);
    files.push(repository.join("Cargo.lock"));
    files.push(repository.join("lamquant-lml-mcu/Cargo.toml"));
    collect_files(&repository.join("lamquant-lml-mcu/src"), &mut files);
    files.sort();

    let mut hasher = blake3::Hasher::new();
    hasher.update(HASH_DOMAIN);
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(repository).unwrap_or(&path);
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        hasher.update(&(relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    let source_id = hasher.finalize().to_hex().to_string();

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_version = command_output(&rustc, &["--version", "--verbose"])
        .lines()
        .collect::<Vec<_>>()
        .join(";");
    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort_unstable();
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned());
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_owned());
    let debug = env::var("DEBUG").unwrap_or_else(|_| "unknown".to_owned());
    let panic_strategy = env::var("CARGO_CFG_PANIC").unwrap_or_else(|_| "unknown".to_owned());
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let build_material = format!(
        "source={source_id};target={target};profile={profile};opt={opt_level};debug={debug};panic={panic_strategy};features={};rustc={rustc_version};rustflags={rustflags}",
        features.join(",")
    );
    let build_id = blake3::hash(build_material.as_bytes()).to_hex().to_string();

    emit("LAMQUANT_ABIR_CODEC_SOURCE_ID", &source_id);
    emit("LAMQUANT_ABIR_CODEC_BUILD_ID", &build_id);
}
