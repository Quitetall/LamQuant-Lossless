use std::fs;
use std::path::Path;

use lamquant_lml_optimum_v2::{EncodeContext, OptimumV2Codec};
use serde_json::json;

const RAW_MAGIC: &[u8; 4] = b"LQR1";
const RAW_HEADER_LEN: usize = 20;
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 8_388_608;
const MAX_LQRAW_BYTES: u64 = (RAW_HEADER_LEN + MAX_VALUES * 4) as u64;

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "LQR1 header is truncated".to_owned())?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| "LQR1 u32 parse failed".to_owned())
}

fn read_lqraw(path: &Path) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
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
    if n_channels == 0
        || n_channels > MAX_CHANNELS
        || n_samples == 0
        || n_samples > MAX_SAMPLES
        || !matches!(
            n_channels.checked_mul(n_samples),
            Some(count) if count <= MAX_VALUES
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

fn run(args: &[String]) -> Result<(), String> {
    if args.len() == 3 && args[1] == "describe" {
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let bytes = fs::read(&executable).map_err(|error| error.to_string())?;
        let descriptor = json!({
            "codec": "LamQuant Optimum v2 native baseline",
            "wire": "LMO1-v3/BGF1-v1",
            "binary": executable,
            "binary_bytes": bytes.len(),
            "package_version": env!("CARGO_PKG_VERSION"),
        });
        return fs::write(
            &args[2],
            serde_json::to_vec_pretty(&descriptor).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", args[2]));
    }
    if args.len() != 4 || !matches!(args[1].as_str(), "encode" | "decode") {
        return Err("usage: optimum-v2-codec encode|decode INPUT OUTPUT | describe OUTPUT".into());
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
