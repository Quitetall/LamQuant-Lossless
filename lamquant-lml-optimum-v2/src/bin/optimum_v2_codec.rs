use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use lamquant_lml_optimum_v2::bgf1_learned::{
    Bgf1ChannelIdentity, Bgf1LearnedCodec, Bgf1LearnedMode,
};
use lamquant_lml_optimum_v2::bgf1_model_pack::{BGF1_EXPECTED_PACK_BYTES, BGF1_MODEL_ID};
use lamquant_lml_optimum_v2::{EncodeContext, OptimumV2Codec};
use serde_json::json;

const RAW_MAGIC: &[u8; 4] = b"LQR1";
const RAW_HEADER_LEN: usize = 20;
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 8_388_608;
const MAX_LQRAW_BYTES: u64 = (RAW_HEADER_LEN + MAX_VALUES * 4) as u64;
const MAX_LEARNED_CHANNELS: usize = 64;
const MAX_LEARNED_VALUES: usize = 131_072;
const MAX_META_BYTES: u64 = 1024 * 1024;
const MAX_LEARNED_PACKET_BYTES: u64 = 64 * 1024 * 1024;
const GOVERNED_CONSTRUCTION_RAW_ROOT: &str =
    "/mnt/4tb/LamQuant/outputs/optimum-v2-development-v2-2k/raw";

fn lexical_absolute(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    Ok(normalized)
}

fn canonical_input_outside_root(input: &Path, denied_root: &Path) -> Result<PathBuf, String> {
    let lexical_input = lexical_absolute(input)?;
    let lexical_root = lexical_absolute(denied_root)?;
    if lexical_input.starts_with(&lexical_root) {
        return Err("learned-encode input is within the governed construction raw root".into());
    }

    let canonical_root = match fs::canonicalize(&lexical_root) {
        Ok(root) => Some(root),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(format!(
                "cannot verify governed construction raw root {}: {error}",
                lexical_root.display()
            ));
        }
    };
    let canonical_input = fs::canonicalize(&lexical_input).map_err(|error| {
        format!(
            "canonicalize learned-encode input {} before open: {error}",
            lexical_input.display()
        )
    })?;
    if canonical_root
        .as_ref()
        .is_some_and(|root| canonical_input.starts_with(root))
    {
        return Err(
            "learned-encode input resolves within the governed construction raw root".into(),
        );
    }
    Ok(canonical_input)
}

fn governed_learned_input(input: &Path) -> Result<PathBuf, String> {
    canonical_input_outside_root(input, Path::new(GOVERNED_CONSTRUCTION_RAW_ROOT))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "LQR1 header is truncated".to_owned())?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| "LQR1 u32 parse failed".to_owned())
}

fn read_lqraw_with_limits(
    path: &Path,
    max_channels: usize,
    max_samples: usize,
    max_values: usize,
) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
    let maximum_bytes = RAW_HEADER_LEN
        .checked_add(
            max_values
                .checked_mul(4)
                .ok_or_else(|| "LQR1 resource bound overflows".to_owned())?,
        )
        .ok_or_else(|| "LQR1 resource bound overflows".to_owned())? as u64;
    let bytes = read_bounded(path, maximum_bytes, "LQR1 input")?;
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
    if n_channels == 0
        || n_channels > max_channels
        || n_samples == 0
        || n_samples > max_samples
        || !matches!(
            n_channels.checked_mul(n_samples),
            Some(count) if count <= max_values
        )
    {
        return Err("LQR1 dimensions exceed resource bounds".into());
    }
    let payload_len = n_channels
        .checked_mul(n_samples)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| "LQR1 dimensions overflow".to_owned())?;
    if bytes.len() != RAW_HEADER_LEN + payload_len {
        return Err("LQR1 payload length does not match dimensions".into());
    }
    let mut signal = Vec::with_capacity(n_channels);
    let mut offset = RAW_HEADER_LEN;
    for _ in 0..n_channels {
        let mut channel = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let sample = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            channel.push(i64::from(sample));
            offset += 4;
        }
        signal.push(channel);
    }
    let labels = std::env::var_os("LQ_CODEC_META_JSON")
        .and_then(|meta| fs::read_to_string(meta).ok())
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| value.get("channel_labels").cloned())
        .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
        .filter(|labels| labels.len() == n_channels)
        .unwrap_or_else(|| (0..n_channels).map(|index| format!("ch{index}")).collect());
    let context = EncodeContext {
        sample_rate_mhz,
        bit_depth,
        channel_labels: labels,
    };
    Ok((signal, context))
}

fn read_lqraw(path: &Path) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
    debug_assert_eq!(MAX_LQRAW_BYTES, (RAW_HEADER_LEN + MAX_VALUES * 4) as u64);
    read_lqraw_with_limits(path, MAX_CHANNELS, MAX_SAMPLES, MAX_VALUES)
}

fn read_learned_lqraw(path: &Path) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
    read_lqraw_with_limits(path, MAX_LEARNED_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES)
}

fn write_lqraw(path: &Path, signal: &[Vec<i64>], context: &EncodeContext) -> Result<(), String> {
    if signal.is_empty()
        || signal[0].is_empty()
        || signal
            .iter()
            .any(|channel| channel.len() != signal[0].len())
    {
        return Err("decoded signal is not rectangular".into());
    }
    let n_channels = u32::try_from(signal.len()).map_err(|_| "too many channels".to_owned())?;
    let n_samples = u32::try_from(signal[0].len()).map_err(|_| "too many samples".to_owned())?;
    let mut bytes = Vec::with_capacity(RAW_HEADER_LEN + signal.len() * signal[0].len() * 4);
    bytes.extend_from_slice(RAW_MAGIC);
    bytes.extend_from_slice(&[1, 4, context.bit_depth, 0]);
    bytes.extend_from_slice(&context.sample_rate_mhz.to_le_bytes());
    bytes.extend_from_slice(&n_channels.to_le_bytes());
    bytes.extend_from_slice(&n_samples.to_le_bytes());
    for channel in signal {
        for &sample in channel {
            let sample =
                i32::try_from(sample).map_err(|_| "decoded sample exceeds i32".to_owned())?;
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn read_bounded(path: &Path, maximum: u64, kind: &str) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let capacity = file
        .metadata()
        .ok()
        .and_then(|metadata| usize::try_from(metadata.len().min(maximum)).ok())
        .unwrap_or(0);
    let read_limit = maximum
        .checked_add(1)
        .ok_or_else(|| format!("{kind} byte bound overflows"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{kind} exceeds its byte bound"));
    }
    Ok(bytes)
}

fn read_learned_model(path: &Path) -> Result<Bgf1LearnedCodec, String> {
    let bytes = read_bounded(path, BGF1_EXPECTED_PACK_BYTES as u64, "BGF1 LQW1 model")?;
    if bytes.len() != BGF1_EXPECTED_PACK_BYTES {
        return Err("BGF1 LQW1 model has the wrong profile length".into());
    }
    Bgf1LearnedCodec::from_lqw1(&bytes).map_err(|error| error.to_string())
}

fn read_learned_identities(
    path: &Path,
    expected_channels: usize,
) -> Result<Vec<Bgf1ChannelIdentity>, String> {
    let bytes = read_bounded(path, MAX_META_BYTES, "BGF1 metadata")?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse metadata: {error}"))?;
    let labels = value
        .get("channel_labels")
        .cloned()
        .ok_or_else(|| "BGF1 metadata is missing channel_labels".to_owned())?;
    let labels: Vec<String> = serde_json::from_value(labels)
        .map_err(|error| format!("parse metadata channel_labels: {error}"))?;
    if labels.len() != expected_channels {
        return Err("BGF1 metadata channel_labels do not match LQR1 channels".into());
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(stable_id, exact_label)| {
            let stable_id = u16::try_from(stable_id)
                .map_err(|_| "BGF1 metadata has too many channels".to_owned())?;
            Ok(Bgf1ChannelIdentity::new(stable_id, exact_label))
        })
        .collect()
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() == 3 && args[1] == "describe" {
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let bytes = fs::read(&executable).map_err(|error| error.to_string())?;
        let descriptor = json!({
            "codec": "LamQuant Optimum v2 native and BGF1 learned carrier",
            "wire": "LMO1-v3/BGF1-v1",
            "binary": executable,
            "binary_bytes": bytes.len(),
            "package_version": env!("CARGO_PKG_VERSION"),
            "learned_worker": {
                "encode": "learned-encode MODE MODEL INPUT META_JSON OUTPUT",
                "decode": "learned-decode MODEL INPUT OUTPUT",
                "modes": [2, 3],
                "model_id": BGF1_MODEL_ID,
            },
        });
        return fs::write(
            &args[2],
            serde_json::to_vec_pretty(&descriptor).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", args[2]));
    }
    if args.len() == 7 && args[1] == "learned-encode" {
        let mode = match args[2].as_str() {
            "2" => Bgf1LearnedMode::NoFlow,
            "3" => Bgf1LearnedMode::Flow,
            _ => return Err("learned-encode MODE must be exactly 2 or 3".into()),
        };
        let input = governed_learned_input(Path::new(&args[4]))?;
        let codec = read_learned_model(Path::new(&args[3]))?;
        let (signal, context) = read_learned_lqraw(&input)?;
        let identities = read_learned_identities(Path::new(&args[5]), signal.len())?;
        let packet = codec
            .encode_window(
                &signal,
                &identities,
                context.sample_rate_mhz,
                context.bit_depth,
                mode,
            )
            .map_err(|error| error.to_string())?;
        return fs::write(&args[6], packet).map_err(|error| format!("write {}: {error}", args[6]));
    }
    if args.len() == 5 && args[1] == "learned-decode" {
        let codec = read_learned_model(Path::new(&args[2]))?;
        let packet = read_bounded(
            Path::new(&args[3]),
            MAX_LEARNED_PACKET_BYTES,
            "BGF1 learned packet",
        )?;
        let decoded = codec
            .decode_window(&packet)
            .map_err(|error| error.to_string())?;
        let context = EncodeContext {
            sample_rate_mhz: decoded.sample_rate_mhz,
            bit_depth: decoded.bit_depth,
            channel_labels: decoded
                .identities
                .iter()
                .map(|identity| identity.exact_label.clone())
                .collect(),
        };
        return write_lqraw(Path::new(&args[4]), &decoded.samples, &context);
    }
    if args.len() != 4 || !matches!(args[1].as_str(), "encode" | "decode") {
        return Err(
            "usage: optimum-v2-codec encode|decode INPUT OUTPUT | describe OUTPUT | learned-encode MODE MODEL INPUT META_JSON OUTPUT | learned-decode MODEL INPUT OUTPUT"
                .into(),
        );
    }
    let codec = OptimumV2Codec;
    let input = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    match args[1].as_str() {
        "encode" => {
            let (signal, context) = read_lqraw(input)?;
            let stream = codec
                .encode_window(&signal, &context)
                .map_err(|error| error.to_string())?;
            fs::write(output, stream)
                .map_err(|error| format!("write {}: {error}", output.display()))
        }
        "decode" => {
            let stream =
                fs::read(input).map_err(|error| format!("read {}: {error}", input.display()))?;
            let decoded = codec
                .decode_window(&stream)
                .map_err(|error| error.to_string())?;
            write_lqraw(output, &decoded.samples, &decoded.context)
        }
        _ => unreachable!(),
    }
}

fn main() {
    if let Err(error) = run(&std::env::args().collect::<Vec<_>>()) {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod governed_raw_guard_tests {
    use super::canonical_input_outside_root;
    use std::fs;

    #[test]
    #[cfg(unix)]
    fn rejects_lexical_descendants_and_canonical_directory_aliases() {
        use std::os::unix::fs::symlink;

        let base =
            std::env::temp_dir().join(format!("optimum_v2_bgf1_path_guard_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let governed_root = base.join("governed/raw");
        fs::create_dir_all(&governed_root).unwrap();
        let governed_input = governed_root.join("input.lqraw");
        fs::write(&governed_input, b"synthetic").unwrap();

        assert!(
            canonical_input_outside_root(&governed_root.join("missing.lqraw"), &governed_root,)
                .is_err()
        );

        let alias = base.join("alias");
        symlink(&governed_root, &alias).unwrap();
        assert!(canonical_input_outside_root(&alias.join("input.lqraw"), &governed_root).is_err());

        let scratch = base.join("scratch.lqraw");
        fs::write(&scratch, b"synthetic").unwrap();
        assert_eq!(
            canonical_input_outside_root(&scratch, &governed_root).unwrap(),
            fs::canonicalize(&scratch).unwrap()
        );
        let _ = fs::remove_dir_all(base);
    }
}
