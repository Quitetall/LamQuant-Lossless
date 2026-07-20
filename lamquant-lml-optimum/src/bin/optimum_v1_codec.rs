use std::fs;
use std::path::{Path, PathBuf};

use lamquant_lml_optimum::{decode_any, Codec, LmoCodec, Mode};
use serde_json::json;
use sha2::{Digest, Sha256};

const RAW_MAGIC: &[u8; 4] = b"LQR1";
const RAW_HEADER_LEN: usize = 20;
const LMO_V2_PREFIX: &[u8; 5] = b"LMO1\x02";
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 8_388_608;
const MAX_LQRAW_BYTES: u64 = (RAW_HEADER_LEN + MAX_VALUES * 4) as u64;
const MAX_LMO_BYTES: u64 = 2 * MAX_LQRAW_BYTES + 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BenchmarkMetadata {
    sample_rate_mhz: u32,
    bit_depth: u8,
    n_channels: usize,
    n_samples: usize,
}

#[derive(Debug)]
struct RawSignal {
    samples: Vec<Vec<i64>>,
    sample_rate_mhz: u32,
    bit_depth: u8,
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "LQR1 header is truncated".to_owned())?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| "LQR1 u32 parse failed".to_owned())
}

fn metadata_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("codec metadata has invalid {field}"))
}

fn metadata_within_bounds(
    sample_rate_mhz: u32,
    bit_depth: u8,
    n_channels: usize,
    n_samples: usize,
) -> bool {
    sample_rate_mhz > 0
        && (1..=32).contains(&bit_depth)
        && (1..=MAX_CHANNELS).contains(&n_channels)
        && (1..=MAX_SAMPLES).contains(&n_samples)
        && matches!(
            n_channels.checked_mul(n_samples),
            Some(count) if count <= MAX_VALUES
        )
}

fn read_metadata() -> Result<BenchmarkMetadata, String> {
    let path = std::env::var_os("LQ_CODEC_META_JSON")
        .map(PathBuf::from)
        .ok_or_else(|| "LQ_CODEC_META_JSON is required".to_owned())?;
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if !value.is_object() {
        return Err("codec metadata must be a JSON object".into());
    }

    let sample_rate_mhz = u32::try_from(metadata_u64(&value, "sample_rate_mhz")?)
        .map_err(|_| "codec metadata has invalid sample_rate_mhz".to_owned())?;
    let bit_depth = u8::try_from(metadata_u64(&value, "bit_depth")?)
        .map_err(|_| "codec metadata has invalid bit_depth".to_owned())?;
    let n_channels = usize::try_from(metadata_u64(&value, "n_channels")?)
        .map_err(|_| "codec metadata has invalid n_channels".to_owned())?;
    let n_samples = usize::try_from(metadata_u64(&value, "n_samples")?)
        .map_err(|_| "codec metadata has invalid n_samples".to_owned())?;
    if !metadata_within_bounds(sample_rate_mhz, bit_depth, n_channels, n_samples) {
        return Err("codec metadata exceeds LQR1 resource bounds".into());
    }
    Ok(BenchmarkMetadata {
        sample_rate_mhz,
        bit_depth,
        n_channels,
        n_samples,
    })
}

fn read_lqraw(path: &Path) -> Result<RawSignal, String> {
    let size = fs::metadata(path)
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if size > MAX_LQRAW_BYTES {
        return Err("LQR1 input exceeds canonical maximum".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() < RAW_HEADER_LEN
        || &bytes[0..4] != RAW_MAGIC
        || bytes[4] != 1
        || bytes[5] != 4
        || bytes[7] != 0
    {
        return Err("unsupported or truncated LQR1 input".into());
    }
    let bit_depth = bytes[6];
    let sample_rate_mhz = read_u32(&bytes, 8)?;
    let n_channels = read_u32(&bytes, 12)? as usize;
    let n_samples = read_u32(&bytes, 16)? as usize;
    if !metadata_within_bounds(sample_rate_mhz, bit_depth, n_channels, n_samples) {
        return Err("LQR1 metadata or dimensions exceed resource bounds".into());
    }
    let payload_len = n_channels
        .checked_mul(n_samples)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| "LQR1 dimensions overflow".to_owned())?;
    if bytes.len() != RAW_HEADER_LEN + payload_len {
        return Err("LQR1 payload length does not match dimensions".into());
    }

    let mut samples = Vec::with_capacity(n_channels);
    let mut offset = RAW_HEADER_LEN;
    for _ in 0..n_channels {
        let mut channel = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let sample = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            channel.push(i64::from(sample));
            offset += 4;
        }
        samples.push(channel);
    }
    Ok(RawSignal {
        samples,
        sample_rate_mhz,
        bit_depth,
    })
}

fn validate_input_metadata(signal: &RawSignal, metadata: BenchmarkMetadata) -> Result<(), String> {
    if signal.sample_rate_mhz != metadata.sample_rate_mhz
        || signal.bit_depth != metadata.bit_depth
        || signal.samples.len() != metadata.n_channels
        || signal.samples[0].len() != metadata.n_samples
    {
        return Err("LQR1 header disagrees with benchmark metadata".into());
    }
    Ok(())
}

fn write_lqraw(
    path: &Path,
    signal: &[Vec<i64>],
    metadata: BenchmarkMetadata,
) -> Result<(), String> {
    if signal.len() != metadata.n_channels
        || signal
            .iter()
            .any(|channel| channel.len() != metadata.n_samples)
    {
        return Err("decoded signal dimensions disagree with benchmark metadata".into());
    }
    let n_channels = u32::try_from(metadata.n_channels).map_err(|_| "too many channels")?;
    let n_samples = u32::try_from(metadata.n_samples).map_err(|_| "too many samples")?;
    let mut bytes = Vec::with_capacity(
        RAW_HEADER_LEN + metadata.n_channels * metadata.n_samples * size_of::<i32>(),
    );
    bytes.extend_from_slice(RAW_MAGIC);
    bytes.extend_from_slice(&[1, 4, metadata.bit_depth, 0]);
    bytes.extend_from_slice(&metadata.sample_rate_mhz.to_le_bytes());
    bytes.extend_from_slice(&n_channels.to_le_bytes());
    bytes.extend_from_slice(&n_samples.to_le_bytes());
    for channel in signal {
        for &sample in channel {
            let sample = i32::try_from(sample).map_err(|_| "decoded sample exceeds signed i32")?;
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn describe(path: &Path) -> Result<(), String> {
    let executable = fs::canonicalize(
        std::env::current_exe().map_err(|error| format!("resolve executable: {error}"))?,
    )
    .map_err(|error| format!("canonicalize executable: {error}"))?;
    let bytes = fs::read(&executable).map_err(|error| format!("read executable: {error}"))?;
    let git_head = env!("LQ_OPTIMUM_GIT_HEAD");
    let git_dirty = env!("LQ_OPTIMUM_GIT_DIRTY");
    let profile = env!("LQ_OPTIMUM_BUILD_PROFILE");
    let target = env!("LQ_OPTIMUM_BUILD_TARGET");
    let features = env!("LQ_OPTIMUM_BUILD_FEATURES")
        .split(',')
        .filter(|feature| !feature.is_empty())
        .collect::<Vec<_>>();
    let rustc_version = env!("LQ_OPTIMUM_RUSTC_VERSION");
    let rustc_commit = env!("LQ_OPTIMUM_RUSTC_COMMIT");
    let build_material = format!(
        "package={}\nversion={}\ngit_head={git_head}\ngit_dirty={git_dirty}\nprofile={profile}\ntarget={target}\nfeatures={}\nrustc_version={rustc_version}\nrustc_commit={rustc_commit}\n",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        features.join(","),
    );
    let build_id = format!("{:x}", Sha256::digest(build_material.as_bytes()));
    let descriptor = json!({
        "codec": "LamQuant Optimum v1 deterministic baseline",
        "wire": "raw LMO1-v2",
        "package": {
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "executable": {
            "path": executable,
            "bytes": bytes.len(),
            "sha256": format!("{:x}", Sha256::digest(&bytes)),
        },
        "source_git": {
            "repository": "codec-lossless",
            "capture": "compile-time",
            "head": (git_head != "unknown").then_some(git_head),
            "dirty": match git_dirty {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            },
        },
        "build": {
            "id": build_id,
            "profile": profile,
            "target": target,
            "features": features,
            "rustc": {
                "version": rustc_version,
                "commit": (rustc_commit != "unknown").then_some(rustc_commit),
            },
        },
    });
    fs::write(
        path,
        serde_json::to_vec_pretty(&descriptor).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))
}

fn invalidate_output(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove stale {}: {error}", path.display())),
    }
}

fn comparable_path(path: &Path) -> Result<PathBuf, String> {
    if let Ok(canonical) = fs::canonicalize(path) {
        return Ok(canonical);
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(canonical_parent) = fs::canonicalize(parent) {
            return Ok(canonical_parent.join(name));
        }
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| format!("resolve {}: {error}", path.display()))
    }
}

fn declared_output(args: &[String]) -> Option<&Path> {
    match args.get(1).map(String::as_str) {
        Some("describe") => args.get(2).map(Path::new),
        Some("encode" | "decode") => args.get(3).map(Path::new),
        _ if args.len() == 4 => Some(Path::new(&args[3])),
        _ => None,
    }
}

fn reject_output_alias(args: &[String], output: &Path) -> Result<(), String> {
    let output = comparable_path(output)?;
    let mut protected = Vec::new();
    let operation = args.get(1).map(String::as_str);
    if matches!(operation, Some("encode" | "decode"))
        || (args.len() == 4 && operation != Some("describe"))
    {
        protected.push(("codec input", PathBuf::from(&args[2])));
        if let Some(metadata) = std::env::var_os("LQ_CODEC_META_JSON") {
            protected.push(("codec metadata", PathBuf::from(metadata)));
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        protected.push(("codec executable", executable));
    }
    for (label, path) in protected {
        if comparable_path(&path)? == output {
            return Err(format!(
                "declared output aliases protected {label}: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let output = declared_output(args);
    if let Some(output) = output {
        reject_output_alias(args, output)?;
        invalidate_output(output)?;
    }
    let result = (|| {
        if args.len() == 3 && args[1] == "describe" {
            return describe(Path::new(&args[2]));
        }
        if args.len() != 4 || !matches!(args[1].as_str(), "encode" | "decode") {
            return Err(
                "usage: optimum-v1-codec encode|decode INPUT OUTPUT | describe OUTPUT".into(),
            );
        }

        let input = Path::new(&args[2]);
        let output = Path::new(&args[3]);
        let metadata = read_metadata()?;
        match args[1].as_str() {
            "encode" => {
                let signal = read_lqraw(input)?;
                validate_input_metadata(&signal, metadata)?;
                let stream = LmoCodec
                    .encode(&signal.samples, Mode::Lossless)
                    .map_err(|error| error.to_string())?;
                if stream.get(..LMO_V2_PREFIX.len()) != Some(LMO_V2_PREFIX) {
                    return Err("Optimum v1 encoder did not emit raw LMO1-v2".into());
                }
                fs::write(output, stream)
                    .map_err(|error| format!("write {}: {error}", output.display()))
            }
            "decode" => {
                let size = fs::metadata(input)
                    .map_err(|error| format!("stat {}: {error}", input.display()))?
                    .len();
                if size > MAX_LMO_BYTES {
                    return Err("LMO1-v2 input exceeds benchmark adapter maximum".into());
                }
                let stream = fs::read(input)
                    .map_err(|error| format!("read {}: {error}", input.display()))?;
                if stream.get(..LMO_V2_PREFIX.len()) != Some(LMO_V2_PREFIX) {
                    return Err("input is not a raw LMO1-v2 stream".into());
                }
                if stream.get(5) != Some(&0) || !matches!(stream.get(6).copied(), Some(0 | 2)) {
                    return Err("input is not an Optimum-v1 lossless stream".into());
                }
                let signal = decode_any(&stream).map_err(|error| error.to_string())?;
                write_lqraw(output, &signal, metadata)
            }
            _ => unreachable!(),
        }
    })();
    if result.is_err() {
        if let Some(output) = output {
            if let Err(cleanup) = invalidate_output(output) {
                return Err(format!(
                    "{}; output cleanup failed: {cleanup}",
                    result.unwrap_err()
                ));
            }
        }
    }
    result
}

fn main() {
    if let Err(error) = run(&std::env::args().collect::<Vec<_>>()) {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
}
