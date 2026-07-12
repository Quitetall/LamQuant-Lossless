use lamquant_core::lma::{self, LmaArchive, Method};
use sha2::{Digest, Sha256};
use std::io::Cursor;

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn legacy_v1_store(path: &str, payload: &[u8]) -> Vec<u8> {
    let manifest = serde_json::json!({
        "compressor": "zstd",
        "files": [{
            "path": path,
            "original_size": payload.len(),
            "compressed_size": payload.len(),
            "method": "store",
            "sha256": sha256(payload),
            "offset": 0,
        }]
    });
    let manifest = serde_json::to_vec(&manifest).unwrap();
    let mut archive = Vec::new();
    archive.extend_from_slice(b"LMA1");
    archive.extend_from_slice(&1u32.to_le_bytes());
    archive.extend_from_slice(&1u32.to_le_bytes());
    archive.extend_from_slice(&((manifest.len() as u32) | 0x8000_0000).to_le_bytes());
    archive.extend_from_slice(&manifest);
    archive.extend_from_slice(payload);
    let digest = Sha256::digest(&archive);
    archive.extend_from_slice(&digest);
    archive
}

fn v2_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    std::fs::create_dir(&input).unwrap();
    std::fs::write(input.join("image.png"), b"stored-payload").unwrap();
    std::fs::write(
        input.join("notes.txt"),
        b"zstd payload with repeated repeated text",
    )
    .unwrap();
    let archive = root.path().join("fixture.lma");
    lma::pack_archive(&input, &archive, 3, false, None).unwrap();
    (root, archive)
}

#[test]
fn v1_borrowed_bytes_adapter_lists_reads_and_verifies() {
    let bytes = legacy_v1_store("legacy.bin", b"legacy-v1-payload");
    let mut archive =
        LmaArchive::from_source(Cursor::new(bytes.as_slice()), bytes.len() as u64).unwrap();

    assert_eq!(archive.entries().len(), 1);
    assert_eq!(archive.entries()[0].method, Method::Store);
    assert_eq!(
        archive.read_raw("legacy.bin").unwrap(),
        b"legacy-v1-payload"
    );
    assert_eq!(
        archive.read_decoded("legacy.bin").unwrap(),
        b"legacy-v1-payload"
    );
    let verification = archive.verify().unwrap();
    assert!(verification.passed());
    assert!(verification.archive_path.is_none());
}

#[test]
fn v2_file_facade_covers_list_read_extract_unpack_append_and_verify() {
    let (root, archive_path) = v2_fixture();
    let mut archive = LmaArchive::open(&archive_path).unwrap();
    assert_eq!(archive.entries().len(), 2);
    assert_eq!(
        archive.read_decoded("image.png").unwrap(),
        b"stored-payload"
    );
    assert_eq!(
        archive.read_decoded("notes.txt").unwrap(),
        b"zstd payload with repeated repeated text"
    );
    assert_eq!(
        archive.read_raw("notes.txt").unwrap(),
        b"zstd payload with repeated repeated text"
    );
    assert!(archive.verify().unwrap().passed());

    let extracted = root.path().join("single.png");
    archive.extract_to("image.png", &extracted).unwrap();
    assert_eq!(std::fs::read(extracted).unwrap(), b"stored-payload");

    let unpacked = root.path().join("unpacked");
    let summary = archive.unpack_to(&unpacked, true, false, None).unwrap();
    assert_eq!(summary.n_files, 2);
    assert_eq!(
        std::fs::read(unpacked.join("image.png")).unwrap(),
        b"stored-payload"
    );

    let extra = root.path().join("extra.png");
    std::fs::write(&extra, b"appended").unwrap();
    drop(archive);
    LmaArchive::append_file(&archive_path, &extra, Some("extra.png"), 3, false, false).unwrap();
    let verification = lma::verify_archive(&archive_path).unwrap();
    assert!(verification.passed());
    assert_eq!(verification.entries.len(), 3);

    // Compatibility adapters remain behavior-preserving while callers migrate.
    assert_eq!(lma::list_archive(&archive_path).unwrap().len(), 3);
    assert_eq!(
        lma::read_entry(&archive_path, "image.png").unwrap(),
        b"stored-payload"
    );
}

#[test]
fn v2_mmap_like_borrowed_bytes_matches_buffered_file_result() {
    let (_root, archive_path) = v2_fixture();
    let bytes = std::fs::read(&archive_path).unwrap();
    let mut borrowed =
        LmaArchive::from_source(Cursor::new(bytes.as_slice()), bytes.len() as u64).unwrap();
    let mut buffered = LmaArchive::open(&archive_path).unwrap();

    assert_eq!(
        borrowed
            .entries()
            .iter()
            .map(|entry| &entry.path)
            .collect::<Vec<_>>(),
        buffered
            .entries()
            .iter()
            .map(|entry| &entry.path)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        borrowed.read_decoded("notes.txt").unwrap(),
        buffered.read_decoded("notes.txt").unwrap()
    );
    assert_eq!(
        borrowed.verify().unwrap().passed(),
        buffered.verify().unwrap().passed()
    );
}

#[test]
fn corruption_returns_one_structured_verification_result() {
    let (_root, archive_path) = v2_fixture();
    let mut bytes = std::fs::read(&archive_path).unwrap();
    bytes[16] ^= 0x40;
    let mut archive =
        LmaArchive::from_source(Cursor::new(bytes.as_slice()), bytes.len() as u64).unwrap();
    let verification = archive.verify().unwrap();

    assert!(!verification.archive_hash_matches);
    assert!(!verification.passed());
    assert!(verification.failed_entries() >= 1);
}

#[test]
fn cli_verifier_contains_no_archive_layout_parser() {
    let cli = include_str!("../src/bin/lml.rs");
    let start = cli.find("fn cmd_verify_archive_explain").unwrap();
    let end = cli[start..].find("fn cmd_encode_bounded_mae").unwrap() + start;
    let verifier = &cli[start..end];
    assert!(verifier.contains("lma::verify_archive(input)"));
    assert!(!verifier.contains("manifest_len"));
    assert!(!verifier.contains("payload_start"));
    assert!(!verifier.contains("SeekFrom"));
    assert!(!verifier.contains("Sha256"));
}

#[test]
fn archive_layout_parser_is_private_to_facade_construction() {
    let archive = include_str!("../src/lma.rs");
    assert_eq!(
        archive.matches("read_lma_index(").count(),
        2, // exactly LmaArchive::open and LmaArchive::from_source
        "only open and from_source may call the parser"
    );
    assert!(archive.contains("fn read_lma_index<R: Read + Seek>("));
    assert!(!archive.contains("pub fn read_lma_index"));
}
