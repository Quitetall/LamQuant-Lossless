use std::collections::BTreeMap;
use std::path::PathBuf;

use lamquant_nodes::verify_lmq_production_authorization;

const MAX_PCCP_BYTES: u64 = 16 * 1024 * 1024;
const MAX_AUTHORIZATION_BYTES: u64 = 4 * 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let pccp = read_bounded_regular_file(required_path(&arguments, "--pccp")?, MAX_PCCP_BYTES)?;
    let authorization = read_bounded_regular_file(
        required_path(&arguments, "--authorization")?,
        MAX_AUTHORIZATION_BYTES,
    )?;
    let registry_sha256 = parse_hex_32(required(&arguments, "--registry-sha256")?)?;
    let checkpoint_sha256 = parse_hex_32(required(&arguments, "--checkpoint-sha256")?)?;
    let verifying_key = parse_hex_32(required(&arguments, "--verifying-key")?)?;
    let verified = verify_lmq_production_authorization(
        &pccp,
        registry_sha256,
        checkpoint_sha256,
        &authorization,
        verifying_key,
    )
    .map_err(|error| format!("production authorization rejected: {error}"))?;
    let (floor_numerator, floor_denominator) = verified.pearson_floor().parts();
    println!(
        "{}",
        serde_json::json!({
            "authorization_epoch": verified.authorization_epoch(),
            "checkpoint_content_id": hex(verified.checkpoint_content_id().as_bytes()),
            "checkpoint_sha256": hex(&verified.checkpoint_sha256()),
            "model_artifact_content_id": hex(verified.model_artifact_content_id().as_bytes()),
            "pccp_change_id": verified.pccp_change_id(),
            "pccp_evidence_id": hex(verified.pccp_evidence_id().as_bytes()),
            "pearson_floor": {
                "denominator": floor_denominator,
                "numerator": floor_numerator,
            },
            "schema": "lamquant.package16-authorization-verification/v1",
            "status": "PASS",
        })
    );
    Ok(())
}

fn parse_arguments() -> Result<BTreeMap<String, String>, String> {
    let values = std::env::args().skip(1).collect::<Vec<_>>();
    if values.len() % 2 != 0 {
        return Err("arguments must be --name value pairs".into());
    }
    let mut arguments = BTreeMap::new();
    for pair in values.chunks_exact(2) {
        if !matches!(
            pair[0].as_str(),
            "--authorization"
                | "--checkpoint-sha256"
                | "--pccp"
                | "--registry-sha256"
                | "--verifying-key"
        ) || arguments.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(format!("unknown or duplicate argument: {}", pair[0]));
        }
    }
    Ok(arguments)
}

fn required<'a>(arguments: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
    arguments
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing argument: {name}"))
}

fn required_path(arguments: &BTreeMap<String, String>, name: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(required(arguments, name)?);
    if !path.is_absolute() {
        return Err(format!("{name} must be absolute"));
    }
    Ok(path)
}

fn read_bounded_regular_file(path: PathBuf, maximum_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = path
        .symlink_metadata()
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > maximum_bytes {
        return Err(format!("invalid bounded regular file: {}", path.display()));
    }
    std::fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("identity must contain 64 hexadecimal digits".into());
    }
    let mut output = [0_u8; 32];
    for (byte, pair) in output.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let pair = std::str::from_utf8(pair).map_err(|_| "identity must be ASCII")?;
        *byte = u8::from_str_radix(pair, 16).map_err(|_| "identity is not hexadecimal")?;
    }
    if hex(&output) != value {
        return Err("identity must use lowercase hexadecimal".into());
    }
    Ok(output)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}
