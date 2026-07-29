use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const HASH_DOMAIN: &[u8] = b"org.quitetall.lamquant.nodes.source-v1\0";

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

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let repository = manifest
        .parent()
        .and_then(Path::parent)
        .expect("node crate must live under codec-lossless/crates");
    let mut files = vec![manifest.join("Cargo.toml"), manifest.join("build.rs")];
    collect_files(&manifest.join("src"), &mut files);
    files.push(repository.join("Cargo.lock"));
    files.push(repository.join("crates/lamquant-abir-codec/Cargo.toml"));
    files.push(repository.join("crates/lamquant-abir-codec/build.rs"));
    collect_files(
        &repository.join("crates/lamquant-abir-codec/src"),
        &mut files,
    );
    files.push(repository.join("lamquant-lml-mcu/Cargo.toml"));
    collect_files(&repository.join("lamquant-lml-mcu/src"), &mut files);
    if env::var_os("CARGO_FEATURE_STD").is_some() {
        files.push(repository.join("lamquant-lml-desktop/Cargo.toml"));
        collect_files(&repository.join("lamquant-lml-desktop/src"), &mut files);
    }
    if env::var_os("CARGO_FEATURE_STANDARD_ADAPTERS").is_some() {
        files.push(repository.join("crates/lamquant-standard-adapters/Cargo.toml"));
        collect_files(
            &repository.join("crates/lamquant-standard-adapters/src"),
            &mut files,
        );
        files.push(repository.join("lamquant-lossless/Cargo.toml"));
        collect_files(&repository.join("lamquant-lossless/src"), &mut files);
    }
    if env::var_os("CARGO_FEATURE_LMQ").is_some() {
        files.push(repository.join("lamquant-lmq/Cargo.toml"));
        collect_files(&repository.join("lamquant-lmq/src"), &mut files);
    }
    files.sort();

    let mut hasher = blake3::Hasher::new();
    hasher.update(HASH_DOMAIN);
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(repository).unwrap_or(&path);
        let relative = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        hasher.update(&(relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    let source_id = hasher.finalize().to_hex();
    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort_unstable();
    println!("cargo:rustc-env=LAMQUANT_NODES_SOURCE_ID={source_id}");
    println!(
        "cargo:rustc-env=LAMQUANT_NODES_FEATURE_SET={}",
        features.join(",")
    );
}
