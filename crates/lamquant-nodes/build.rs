use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const HASH_DOMAIN: &[u8] = b"org.quitetall.lamquant.nodes.source-v1\0";
const IMPLEMENTATION_DOMAIN: &[u8] = b"org.quitetall.lamquant.nodes.implementation-v1\0";
const MCU_FEATURE_SET: &str = "mcu-aot-baseline";
const MCU_FUSED_LOWERING: &str = "fused:org.quitetall.lamquant.lml.reference-v1";

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

fn hash_source(repository: &Path, files: &[PathBuf]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(HASH_DOMAIN);
    for path in files {
        let relative = path.strip_prefix(repository).unwrap_or(path);
        let relative = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        hasher.update(&(relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    hasher.finalize()
}

fn implementation_id(
    source_id: &str,
    feature_set: &str,
    lowering: &str,
    target: u8,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(IMPLEMENTATION_DOMAIN);
    hasher.update(source_id.as_bytes());
    hasher.update(&[0]);
    hasher.update(feature_set.as_bytes());
    hasher.update(&[0]);
    hasher.update(lowering.as_bytes());
    hasher.update(&[0, target]);
    hasher.finalize()
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let repository = manifest
        .parent()
        .and_then(Path::parent)
        .expect("node crate must live under codec-lossless/crates");
    let mut mcu_files = vec![manifest.join("Cargo.toml"), manifest.join("build.rs")];
    collect_files(&manifest.join("src"), &mut mcu_files);
    mcu_files.push(repository.join("Cargo.lock"));
    mcu_files.push(repository.join("crates/lamquant-abir-codec/Cargo.toml"));
    mcu_files.push(repository.join("crates/lamquant-abir-codec/build.rs"));
    collect_files(
        &repository.join("crates/lamquant-abir-codec/src"),
        &mut mcu_files,
    );
    mcu_files.push(repository.join("crates/lamquant-abir-montage/Cargo.toml"));
    collect_files(
        &repository.join("crates/lamquant-abir-montage/src"),
        &mut mcu_files,
    );
    mcu_files.push(repository.join("crates/lamquant-common/Cargo.toml"));
    collect_files(
        &repository.join("crates/lamquant-common/src"),
        &mut mcu_files,
    );
    mcu_files.push(repository.join("lamquant-lml-mcu/Cargo.toml"));
    collect_files(&repository.join("lamquant-lml-mcu/src"), &mut mcu_files);
    mcu_files.sort();
    let mut files = mcu_files.clone();
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
    if env::var_os("CARGO_FEATURE_OPTIMUM_V2").is_some() {
        files.push(repository.join("lamquant-lml-optimum-v2/Cargo.toml"));
        files.push(repository.join("lamquant-lml-optimum-v2/build.rs"));
        collect_files(&repository.join("lamquant-lml-optimum-v2/src"), &mut files);
    }
    if env::var_os("CARGO_FEATURE_LMQ").is_some() {
        files.push(repository.join("lamquant-lmq/Cargo.toml"));
        collect_files(&repository.join("lamquant-lmq/src"), &mut files);
    }
    files.sort();

    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let source_id = hash_source(repository, &files).to_hex();
    let mcu_source_id = hash_source(repository, &mcu_files).to_hex();
    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort_unstable();
    let feature_set = features.join(",");
    println!("cargo:rustc-env=LAMQUANT_NODES_SOURCE_ID={source_id}");
    println!("cargo:rustc-env=LAMQUANT_NODES_MCU_SOURCE_ID={mcu_source_id}");
    println!("cargo:rustc-env=LAMQUANT_NODES_FEATURE_SET={feature_set}");

    let mcu_fused = implementation_id(&mcu_source_id, MCU_FEATURE_SET, MCU_FUSED_LOWERING, 0);
    let bytes = mcu_fused
        .as_bytes()
        .iter()
        .map(|byte| format!("0x{byte:02x}"))
        .collect::<Vec<_>>()
        .join(", ");
    let generated = format!(
        "pub(crate) const REFERENCE_FUSED_MCU_IMPLEMENTATION_BYTES: [u8; 32] = [{bytes}];\n"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"))
        .join("mcu_implementation_ids.rs");
    fs::write(&output, generated)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", output.display()));
}
