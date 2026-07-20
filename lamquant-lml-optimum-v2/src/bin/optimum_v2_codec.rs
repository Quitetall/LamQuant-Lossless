#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use lamquant_lml_optimum_v2::bgf1_learned::{
    Bgf1ChannelIdentity, Bgf1LearnedCodec, Bgf1LearnedMode,
};
use lamquant_lml_optimum_v2::bgf1_model_pack::{BGF1_EXPECTED_PACK_BYTES, BGF1_MODEL_ID};
use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix1_carrier::{Dix1CarrierMode, Dix1ConstructionCodec};
use lamquant_lml_optimum_v2::dix2_carrier::{Dix2CarrierMode, Dix2ConstructionCodec};
use lamquant_lml_optimum_v2::mix1::{Mix1Codec, Mix1EntropyProfile, Mix1TunedProfile};
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

fn absolute_unresolved(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path))
    }
}

fn lexical_normalize(absolute: &Path) -> PathBuf {
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
    normalized
}

struct GovernedRawBoundary {
    lexical_root: PathBuf,
    canonical_root: Option<PathBuf>,
}

struct GovernedOutputPath {
    #[cfg(target_os = "linux")]
    expected_parent: PathBuf,
    #[cfg(target_os = "linux")]
    name: std::ffi::OsString,
    #[cfg(target_os = "linux")]
    parent: File,
    path: PathBuf,
}

impl GovernedOutputPath {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl GovernedRawBoundary {
    fn new(root: &Path) -> Result<Self, String> {
        let absolute_root = absolute_unresolved(root)?;
        let lexical_root = lexical_normalize(&absolute_root);
        let canonical_root = match fs::canonicalize(&absolute_root) {
            Ok(root) => Some(root),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(format!(
                    "cannot verify governed construction raw root {}: {error}",
                    lexical_root.display()
                ));
            }
        };
        Ok(Self {
            lexical_root,
            canonical_root,
        })
    }

    fn fixed() -> Result<Self, String> {
        Self::new(Path::new(GOVERNED_CONSTRUCTION_RAW_ROOT))
    }

    fn reject_lexical(&self, path: &Path, operand: &str) -> Result<(), String> {
        if path.starts_with(&self.lexical_root) {
            return Err(format!(
                "construction worker {operand} is within the governed construction raw root"
            ));
        }
        Ok(())
    }

    fn reject_canonical(&self, path: &Path, operand: &str) -> Result<(), String> {
        if self
            .canonical_root
            .as_ref()
            .is_some_and(|root| path.starts_with(root))
        {
            return Err(format!(
                "construction worker {operand} resolves within the governed construction raw root"
            ));
        }
        Ok(())
    }

    fn existing_path(&self, path: &Path, operand: &str) -> Result<PathBuf, String> {
        let absolute = absolute_unresolved(path)?;
        self.reject_lexical(&lexical_normalize(&absolute), operand)?;
        let canonical = fs::canonicalize(&absolute).map_err(|error| {
            format!(
                "canonicalize construction worker {operand} {} before open: {error}",
                absolute.display()
            )
        })?;
        self.reject_canonical(&canonical, operand)?;
        Ok(canonical)
    }

    #[cfg(target_os = "linux")]
    fn output_path(&self, path: &Path) -> Result<GovernedOutputPath, String> {
        let absolute = absolute_unresolved(path)?;
        self.reject_lexical(&lexical_normalize(&absolute), "OUTPUT")?;
        let parent_path = absolute
            .parent()
            .ok_or_else(|| "construction worker OUTPUT has no parent".to_owned())?;
        let name = absolute
            .file_name()
            .ok_or_else(|| "construction worker OUTPUT has no file name".to_owned())?
            .to_owned();
        let parent = File::open(parent_path).map_err(|error| {
            format!(
                "open construction worker OUTPUT parent {}: {error}",
                parent_path.display()
            )
        })?;
        let descriptor_parent = PathBuf::from(format!("/proc/self/fd/{}", parent.as_raw_fd()));
        let expected_parent = fs::canonicalize(&descriptor_parent).map_err(|error| {
            format!(
                "resolve opened construction worker OUTPUT parent {}: {error}",
                parent_path.display()
            )
        })?;
        self.reject_canonical(&expected_parent, "OUTPUT")?;
        if fs::canonicalize(parent_path).map_err(|error| {
            format!(
                "revalidate construction worker OUTPUT parent {}: {error}",
                parent_path.display()
            )
        })? != expected_parent
        {
            return Err("construction worker OUTPUT parent changed during validation".to_owned());
        }
        Ok(GovernedOutputPath {
            path: expected_parent.join(&name),
            expected_parent,
            name,
            parent,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn output_path(&self, _path: &Path) -> Result<GovernedOutputPath, String> {
        Err("construction worker OUTPUT requires Linux descriptor-bound creation".to_owned())
    }

    fn verify_opened(&self, file: &File, expected: &Path, operand: &str) -> Result<(), String> {
        let metadata = file
            .metadata()
            .map_err(|error| format!("inspect opened construction worker {operand}: {error}"))?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "construction worker {operand} is not a regular file"
            ));
        }
        #[cfg(unix)]
        if metadata.nlink() != 1 {
            return Err(format!(
                "construction worker {operand} is hard-linked; governed boundary requires one link"
            ));
        }
        #[cfg(target_os = "linux")]
        {
            let opened = fs::canonicalize(format!("/proc/self/fd/{}", file.as_raw_fd())).map_err(
                |error| {
                    format!("resolve opened construction worker {operand} before access: {error}")
                },
            )?;
            if opened != expected {
                return Err(format!(
                    "construction worker {operand} changed between validation and open"
                ));
            }
            self.reject_canonical(&opened, operand)?;
        }
        Ok(())
    }

    fn open_existing(&self, path: &Path, operand: &str) -> Result<File, String> {
        #[cfg(target_os = "linux")]
        let file = {
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
            options.open(path)
        };
        #[cfg(not(target_os = "linux"))]
        let file = File::open(path);
        let file = file.map_err(|error| {
            format!(
                "open construction worker {operand} {}: {error}",
                path.display()
            )
        })?;
        self.verify_opened(&file, path, operand)?;
        Ok(file)
    }

    #[cfg(target_os = "linux")]
    fn open_output(&self, output: &GovernedOutputPath) -> Result<File, String> {
        let descriptor_parent =
            PathBuf::from(format!("/proc/self/fd/{}", output.parent.as_raw_fd()));
        if fs::canonicalize(&descriptor_parent).map_err(|error| {
            format!(
                "resolve opened construction worker OUTPUT parent {}: {error}",
                output.expected_parent.display()
            )
        })? != output.expected_parent
        {
            return Err("construction worker OUTPUT parent changed before creation".to_owned());
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        options.mode(0o600);
        let file = options
            .open(descriptor_parent.join(&output.name))
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    format!(
                        "refuse existing construction worker OUTPUT {}",
                        output.path.display()
                    )
                } else {
                    format!(
                        "open construction worker OUTPUT {}: {error}",
                        output.path.display()
                    )
                }
            })?;
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| {
                format!(
                    "set construction worker OUTPUT mode {}: {error}",
                    output.path.display()
                )
            })?;
        self.verify_opened(&file, &output.path, "OUTPUT")?;
        if file
            .metadata()
            .map_err(|error| format!("inspect construction worker OUTPUT mode: {error}"))?
            .permissions()
            .mode()
            & 0o777
            != 0o600
        {
            return Err("construction worker OUTPUT mode is not exactly 0600".to_owned());
        }
        Ok(file)
    }

    #[cfg(not(target_os = "linux"))]
    fn open_output(&self, _output: &GovernedOutputPath) -> Result<File, String> {
        Err("construction worker OUTPUT requires Linux descriptor-bound creation".to_owned())
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "LQR1 header is truncated".to_owned())?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| "LQR1 u32 parse failed".to_owned())
}

fn lqraw_maximum_bytes(max_values: usize) -> Result<u64, String> {
    RAW_HEADER_LEN
        .checked_add(
            max_values
                .checked_mul(4)
                .ok_or_else(|| "LQR1 resource bound overflows".to_owned())?,
        )
        .map(|maximum| maximum as u64)
        .ok_or_else(|| "LQR1 resource bound overflows".to_owned())
}

fn parse_lqraw_with_limits(
    bytes: Vec<u8>,
    max_channels: usize,
    max_samples: usize,
    max_values: usize,
    read_environment_labels: bool,
) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
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
    let labels = read_environment_labels
        .then(|| std::env::var_os("LQ_CODEC_META_JSON"))
        .flatten()
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
    let bytes = read_bounded(path, lqraw_maximum_bytes(MAX_VALUES)?, "LQR1 input")?;
    parse_lqraw_with_limits(bytes, MAX_CHANNELS, MAX_SAMPLES, MAX_VALUES, true)
}

fn read_learned_lqraw_file(
    file: File,
    path: &Path,
) -> Result<(Vec<Vec<i64>>, EncodeContext), String> {
    let bytes = read_bounded_file(
        file,
        path,
        lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
        "LQR1 input",
    )?;
    parse_lqraw_with_limits(
        bytes,
        MAX_LEARNED_CHANNELS,
        MAX_SAMPLES,
        MAX_LEARNED_VALUES,
        false,
    )
}

fn write_lqraw(path: &Path, signal: &[Vec<i64>], context: &EncodeContext) -> Result<(), String> {
    fs::write(path, encode_lqraw(signal, context)?)
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn read_bounded(path: &Path, maximum: u64, kind: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    read_bounded_file(file, path, maximum, kind)
}

fn read_bounded_file(
    mut file: File,
    path: &Path,
    maximum: u64,
    kind: &str,
) -> Result<Vec<u8>, String> {
    let capacity = file
        .metadata()
        .ok()
        .and_then(|metadata| usize::try_from(metadata.len().min(maximum)).ok())
        .unwrap_or(0);
    let read_limit = maximum
        .checked_add(1)
        .ok_or_else(|| format!("{kind} byte bound overflows"))?;
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(&mut file)
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
    parse_learned_model(bytes)
}

fn read_learned_model_file(file: File, path: &Path) -> Result<Bgf1LearnedCodec, String> {
    let bytes = read_bounded_file(
        file,
        path,
        BGF1_EXPECTED_PACK_BYTES as u64,
        "BGF1 LQW1 model",
    )?;
    parse_learned_model(bytes)
}

fn parse_learned_model(bytes: Vec<u8>) -> Result<Bgf1LearnedCodec, String> {
    if bytes.len() != BGF1_EXPECTED_PACK_BYTES {
        return Err("BGF1 LQW1 model has the wrong profile length".into());
    }
    Bgf1LearnedCodec::from_lqw1(&bytes).map_err(|error| error.to_string())
}

fn read_learned_identities(
    file: File,
    path: &Path,
    expected_channels: usize,
) -> Result<Vec<Bgf1ChannelIdentity>, String> {
    let bytes = read_bounded_file(file, path, MAX_META_BYTES, "BGF1 metadata")?;
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

fn read_dix1_identities(
    file: File,
    path: &Path,
    expected_channels: usize,
) -> Result<Vec<ChannelIdentity>, String> {
    let bytes = read_bounded_file(file, path, MAX_META_BYTES, "DIX1 metadata")?;
    parse_dix1_identities(&bytes, expected_channels)
}

fn parse_dix1_identities(
    bytes: &[u8],
    expected_channels: usize,
) -> Result<Vec<ChannelIdentity>, String> {
    if bytes.len() as u64 > MAX_META_BYTES {
        return Err("DIX1 metadata exceeds its byte bound".to_owned());
    }
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("parse metadata: {error}"))?;
    let labels = value
        .get("channel_labels")
        .cloned()
        .ok_or_else(|| "DIX1 metadata is missing channel_labels".to_owned())?;
    let labels: Vec<String> = serde_json::from_value(labels)
        .map_err(|error| format!("parse metadata channel_labels: {error}"))?;
    if labels.len() != expected_channels {
        return Err("DIX1 metadata channel_labels do not match LQR1 channels".into());
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(stable_id, exact_label)| {
            let stable_id = u16::try_from(stable_id)
                .map_err(|_| "DIX1 metadata has too many channels".to_owned())?;
            Ok(ChannelIdentity::new(stable_id, exact_label))
        })
        .collect()
}

fn encode_dix1_packet(
    profile: &str,
    signal: &[Vec<i64>],
    identities: &[ChannelIdentity],
    context: &EncodeContext,
) -> Result<Vec<u8>, String> {
    let codec = Dix1ConstructionCodec;
    match profile {
        "product" => codec.encode_window(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
        ),
        "native" => codec.encode_native_window(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
        ),
        "raw" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix1CarrierMode::Raw,
        ),
        "delta" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix1CarrierMode::Delta,
        ),
        "incidence" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix1CarrierMode::IncidenceRans,
        ),
        "no-incidence" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix1CarrierMode::NoIncidenceRans,
        ),
        _ => {
            return Err(
                "DIX1 PROFILE must be product, native, raw, delta, incidence, or no-incidence"
                    .to_owned(),
            );
        }
    }
    .map_err(|error| error.to_string())
}

fn encode_dix2_packet(
    profile: &str,
    signal: &[Vec<i64>],
    identities: &[ChannelIdentity],
    context: &EncodeContext,
) -> Result<Vec<u8>, String> {
    let codec = Dix2ConstructionCodec;
    match profile {
        "product" => codec.encode_window(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
        ),
        "native" => codec.encode_native_window(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
        ),
        "raw" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix2CarrierMode::Raw,
        ),
        "delta" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix2CarrierMode::Delta,
        ),
        "temporal" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix2CarrierMode::TemporalRans,
        ),
        "tree" => codec.encode_forced(
            signal,
            identities,
            context.sample_rate_mhz,
            context.bit_depth,
            Dix2CarrierMode::TreeMedRans,
        ),
        _ => {
            return Err(
                "DIX2 PROFILE must be product, native, raw, delta, temporal, or tree".to_owned(),
            );
        }
    }
    .map_err(|error| error.to_string())
}

fn read_standard_input(maximum: u64, kind: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(
            maximum
                .checked_add(1)
                .ok_or_else(|| format!("{kind} byte bound overflows"))?,
        )
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {kind} from standard input: {error}"))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{kind} exceeds its byte bound"));
    }
    Ok(bytes)
}

fn write_standard_output(bytes: &[u8]) -> Result<(), String> {
    let mut output = std::io::stdout().lock();
    output
        .write_all(bytes)
        .and_then(|()| output.flush())
        .map_err(|error| format!("write construction worker standard output: {error}"))
}

fn write_governed_bytes(
    boundary: &GovernedRawBoundary,
    output: &GovernedOutputPath,
    bytes: &[u8],
) -> Result<(), String> {
    let mut output_file = boundary.open_output(output)?;
    output_file
        .write_all(bytes)
        .map_err(|error| format!("write {}: {error}", output.path().display()))
}

fn encode_lqraw(signal: &[Vec<i64>], context: &EncodeContext) -> Result<Vec<u8>, String> {
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
    Ok(bytes)
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() == 3 && args[1] == "describe" {
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let bytes = fs::read(&executable).map_err(|error| error.to_string())?;
        let descriptor = json!({
            "codec": "LamQuant Optimum v2 native, MIX1, DIX1/DIX2 construction, and BGF1 learned carrier",
            "wire": "LMO1-v3/BGF1-v1/OV2P-v2-v4-MIX1-ALX1/DIX1-v2/DIX2-v1-construction",
            "binary": executable,
            "binary_bytes": bytes.len(),
            "package_version": env!("CARGO_PKG_VERSION"),
            "learned_worker": {
                "encode": "learned-encode MODE MODEL INPUT META_JSON OUTPUT",
                "decode": "learned-decode MODEL INPUT OUTPUT",
                "modes": [2, 3],
                "model_id": BGF1_MODEL_ID,
            },
            "dix1_worker": {
                "encode": "dix1-encode PROFILE INPUT META_JSON OUTPUT",
                "decode": "dix1-decode INPUT OUTPUT",
                "encode_stdio": "dix1-encode-stdio PROFILE META_JSON",
                "decode_stdio": "dix1-decode-stdio",
                "profiles": [
                    "product",
                    "native",
                    "raw",
                    "delta",
                    "incidence",
                    "no-incidence",
                ],
                "body_version": 2,
                "construction_private": true,
            },
            "dix2_worker": {
                "encode_stdio": "dix2-encode-stdio PROFILE META_JSON",
                "decode_stdio": "dix2-decode-stdio",
                "profiles": [
                    "product",
                    "native",
                    "raw",
                    "delta",
                    "temporal",
                    "tree",
                ],
                "body_version": 1,
                "construction_private": true,
            },
            "mix1_worker": {
                "encode_stdio": "mix1-encode-stdio SCORE_SHIFT",
                "encode_best_stdio": "mix1-encode-best-stdio",
                "encode_peer_best_stdio": "mix1-peer-encode-best-stdio",
                "encode_peer_best_no_alias_stdio": "mix1-peer-encode-best-no-alias-stdio",
                "encode_peer_permuted_stdio": "mix1-peer-permuted-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK",
                "encode_peer_tuned_stdio": "mix1-peer-tuned-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK HISTORY_CONTEXT SCALE_PROFILE PARENT_HISTORY_DEPTH PARENT_PENALTY",
                "encode_peer_compact_common_profile_stdio": "mix1-peer-compact-common-profile-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK HISTORY_CONTEXT SCALE_PROFILE",
                "decode_stdio": "mix1-decode-stdio",
                "score_shifts": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "channel_context_masks": [2, 3, 4, 5, 6, 7],
                "peer_magics": ["MIX1", "MMV1", "MCH1", "MCX1", "MQX1", "MPX1", "APX1", "BQX1", "ALX1"],
                "development_only": true,
            },
        });
        return fs::write(
            &args[2],
            serde_json::to_vec_pretty(&descriptor).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", args[2]));
    }
    if args.len() == 4 && args[1] == "dix1-encode-stdio" {
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "DIX1 LQR1 standard input",
        )?;
        let (signal, context) = parse_lqraw_with_limits(
            raw,
            MAX_LEARNED_CHANNELS,
            MAX_SAMPLES,
            MAX_LEARNED_VALUES,
            false,
        )?;
        let identities = parse_dix1_identities(args[3].as_bytes(), signal.len())?;
        let packet = encode_dix1_packet(&args[2], &signal, &identities, &context)?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "dix1-decode-stdio" {
        let packet = read_standard_input(MAX_LEARNED_PACKET_BYTES, "DIX1 packet standard input")?;
        let decoded = Dix1ConstructionCodec
            .decode_window(&packet)
            .map_err(|error| error.to_string())?;
        let context = EncodeContext {
            sample_rate_mhz: decoded.sample_rate_mhz,
            bit_depth: decoded.bit_depth,
            channel_labels: decoded
                .identities
                .iter()
                .map(|identity| identity.label.clone())
                .collect(),
        };
        return write_standard_output(&encode_lqraw(&decoded.samples, &context)?);
    }
    if args.len() == 4 && args[1] == "dix2-encode-stdio" {
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "DIX2 LQR1 standard input",
        )?;
        let (signal, context) = parse_lqraw_with_limits(
            raw,
            MAX_LEARNED_CHANNELS,
            MAX_SAMPLES,
            MAX_LEARNED_VALUES,
            false,
        )?;
        let identities = parse_dix1_identities(args[3].as_bytes(), signal.len())?;
        let packet = encode_dix2_packet(&args[2], &signal, &identities, &context)?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "dix2-decode-stdio" {
        let packet = read_standard_input(MAX_LEARNED_PACKET_BYTES, "DIX2 packet standard input")?;
        let decoded = Dix2ConstructionCodec
            .decode_window(&packet)
            .map_err(|error| error.to_string())?;
        let context = EncodeContext {
            sample_rate_mhz: decoded.sample_rate_mhz,
            bit_depth: decoded.bit_depth,
            channel_labels: decoded
                .identities
                .iter()
                .map(|identity| identity.label.clone())
                .collect(),
        };
        return write_standard_output(&encode_lqraw(&decoded.samples, &context)?);
    }
    if args.len() == 3 && args[1] == "mix1-encode-stdio" {
        let score_shift: u8 = args[2]
            .parse()
            .map_err(|_| "MIX1 SCORE_SHIFT must be an integer in 2..=8".to_owned())?;
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX1 LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_window(
                &signal,
                context.sample_rate_mhz,
                context.bit_depth,
                score_shift,
            )
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "mix1-encode-best-stdio" {
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX1 LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_best_score_window(&signal, context.sample_rate_mhz, context.bit_depth)
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "mix1-peer-encode-best-stdio" {
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX peer LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_best_peer_window(&signal, context.sample_rate_mhz, context.bit_depth)
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "mix1-peer-encode-best-no-alias-stdio" {
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX peer control LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_best_peer_window_without_alias(
                &signal,
                context.sample_rate_mhz,
                context.bit_depth,
            )
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 4 && args[1] == "mix1-peer-permuted-encode-stdio" {
        let score_shift: u8 = args[2]
            .parse()
            .map_err(|_| "MIX peer SCORE_SHIFT must be an integer in 2..=8".to_owned())?;
        let channel_context_mask: u8 = args[3]
            .parse()
            .map_err(|_| "MIX peer CHANNEL_CONTEXT_MASK must be an integer in 2..=7".to_owned())?;
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX peer permuted LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_permuted_common_mode_window(
                &signal,
                context.sample_rate_mhz,
                context.bit_depth,
                score_shift,
                channel_context_mask,
            )
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 8 && args[1] == "mix1-peer-tuned-encode-stdio" {
        let score_shift: u8 = args[2]
            .parse()
            .map_err(|_| "MIX peer SCORE_SHIFT must be an integer in 2..=8".to_owned())?;
        let channel_context_mask: u8 = args[3]
            .parse()
            .map_err(|_| "MIX peer CHANNEL_CONTEXT_MASK must be an integer in 2..=7".to_owned())?;
        let history_context: u8 = args[4]
            .parse()
            .map_err(|_| "MIX peer HISTORY_CONTEXT must be an integer in 0..=15".to_owned())?;
        let scale_profile: u8 = args[5]
            .parse()
            .map_err(|_| "MIX peer SCALE_PROFILE must be an integer in 0..=6".to_owned())?;
        let parent_history_depth: u8 = args[6]
            .parse()
            .map_err(|_| "MIX peer PARENT_HISTORY_DEPTH must be an integer in 0..=4".to_owned())?;
        let parent_penalty: u64 = args[7]
            .parse()
            .map_err(|_| "MIX peer PARENT_PENALTY must be a nonnegative integer".to_owned())?;
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX peer tuned-profile LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_tuned_permuted_window(
                &signal,
                context.sample_rate_mhz,
                context.bit_depth,
                Mix1TunedProfile {
                    entropy: Mix1EntropyProfile {
                        score_shift,
                        channel_context_mask,
                        history_context,
                        scale_profile,
                    },
                    parent_history_depth,
                    parent_penalty,
                },
            )
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 6 && args[1] == "mix1-peer-compact-common-profile-encode-stdio" {
        let score_shift: u8 = args[2]
            .parse()
            .map_err(|_| "MIX peer SCORE_SHIFT must be an integer in 2..=12".to_owned())?;
        let channel_context_mask: u8 = args[3]
            .parse()
            .map_err(|_| "MIX peer CHANNEL_CONTEXT_MASK must be an integer in 2..=7".to_owned())?;
        let history_context: u8 = args[4]
            .parse()
            .map_err(|_| "MIX peer HISTORY_CONTEXT must be a nonzero byte".to_owned())?;
        let scale_profile: u8 = args[5]
            .parse()
            .map_err(|_| "MIX peer SCALE_PROFILE must be an integer in 0..=6".to_owned())?;
        let raw = read_standard_input(
            lqraw_maximum_bytes(MAX_LEARNED_VALUES)?,
            "MIX peer compact common-mode profile LQR1 standard input",
        )?;
        let (signal, context) =
            parse_lqraw_with_limits(raw, MAX_CHANNELS, MAX_SAMPLES, MAX_LEARNED_VALUES, false)?;
        let packet = Mix1Codec
            .encode_compact_common_profile_window(
                &signal,
                context.sample_rate_mhz,
                context.bit_depth,
                Mix1EntropyProfile {
                    score_shift,
                    channel_context_mask,
                    history_context,
                    scale_profile,
                },
            )
            .map_err(|error| error.to_string())?;
        return write_standard_output(&packet);
    }
    if args.len() == 2 && args[1] == "mix1-decode-stdio" {
        let packet = read_standard_input(MAX_LEARNED_PACKET_BYTES, "MIX1 packet standard input")?;
        let decoded = Mix1Codec
            .decode_window(&packet)
            .map_err(|error| error.to_string())?;
        let context = EncodeContext {
            sample_rate_mhz: decoded.sample_rate_mhz,
            bit_depth: decoded.bit_depth,
            channel_labels: Vec::new(),
        };
        return write_standard_output(&encode_lqraw(&decoded.samples, &context)?);
    }
    if args.len() == 6 && args[1] == "dix1-encode" {
        let profile = args[2].as_str();
        if !matches!(
            profile,
            "product" | "native" | "raw" | "delta" | "incidence" | "no-incidence"
        ) {
            return Err(
                "dix1-encode PROFILE must be product, native, raw, delta, incidence, or no-incidence"
                    .into(),
            );
        }
        let boundary = GovernedRawBoundary::fixed()?;
        let input = boundary.existing_path(Path::new(&args[3]), "INPUT")?;
        let metadata = boundary.existing_path(Path::new(&args[4]), "META_JSON")?;
        let output = boundary.output_path(Path::new(&args[5]))?;
        let input_file = boundary.open_existing(&input, "INPUT")?;
        let metadata_file = boundary.open_existing(&metadata, "META_JSON")?;
        let (signal, context) = read_learned_lqraw_file(input_file, &input)?;
        let identities = read_dix1_identities(metadata_file, &metadata, signal.len())?;
        let packet = encode_dix1_packet(profile, &signal, &identities, &context)?;
        return write_governed_bytes(&boundary, &output, &packet);
    }
    if args.len() == 4 && args[1] == "dix1-decode" {
        let boundary = GovernedRawBoundary::fixed()?;
        let input = boundary.existing_path(Path::new(&args[2]), "INPUT")?;
        let output = boundary.output_path(Path::new(&args[3]))?;
        let input_file = boundary.open_existing(&input, "INPUT")?;
        let packet =
            read_bounded_file(input_file, &input, MAX_LEARNED_PACKET_BYTES, "DIX1 packet")?;
        let decoded = Dix1ConstructionCodec
            .decode_window(&packet)
            .map_err(|error| error.to_string())?;
        let context = EncodeContext {
            sample_rate_mhz: decoded.sample_rate_mhz,
            bit_depth: decoded.bit_depth,
            channel_labels: decoded
                .identities
                .iter()
                .map(|identity| identity.label.clone())
                .collect(),
        };
        let raw = encode_lqraw(&decoded.samples, &context)?;
        return write_governed_bytes(&boundary, &output, &raw);
    }
    if args.len() == 7 && args[1] == "learned-encode" {
        let mode = match args[2].as_str() {
            "2" => Bgf1LearnedMode::NoFlow,
            "3" => Bgf1LearnedMode::Flow,
            _ => return Err("learned-encode MODE must be exactly 2 or 3".into()),
        };
        let boundary = GovernedRawBoundary::fixed()?;
        let input = boundary.existing_path(Path::new(&args[4]), "INPUT")?;
        let model = boundary.existing_path(Path::new(&args[3]), "MODEL")?;
        let metadata = boundary.existing_path(Path::new(&args[5]), "META_JSON")?;
        let output = boundary.output_path(Path::new(&args[6]))?;
        let input_file = boundary.open_existing(&input, "INPUT")?;
        let model_file = boundary.open_existing(&model, "MODEL")?;
        let metadata_file = boundary.open_existing(&metadata, "META_JSON")?;
        let codec = read_learned_model_file(model_file, &model)?;
        let (signal, context) = read_learned_lqraw_file(input_file, &input)?;
        let identities = read_learned_identities(metadata_file, &metadata, signal.len())?;
        let packet = codec
            .encode_window(
                &signal,
                &identities,
                context.sample_rate_mhz,
                context.bit_depth,
                mode,
            )
            .map_err(|error| error.to_string())?;
        let mut output_file = boundary.open_output(&output)?;
        output_file
            .set_len(0)
            .map_err(|error| format!("truncate {}: {error}", output.path().display()))?;
        return output_file
            .write_all(&packet)
            .map_err(|error| format!("write {}: {error}", output.path().display()));
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
            "usage: optimum-v2-codec encode|decode INPUT OUTPUT | describe OUTPUT | mix1-encode-stdio SCORE_SHIFT | mix1-encode-best-stdio | mix1-peer-encode-best-stdio | mix1-peer-encode-best-no-alias-stdio | mix1-peer-permuted-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK | mix1-peer-tuned-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK HISTORY_CONTEXT SCALE_PROFILE PARENT_HISTORY_DEPTH PARENT_PENALTY | mix1-peer-compact-common-profile-encode-stdio SCORE_SHIFT CHANNEL_CONTEXT_MASK HISTORY_CONTEXT SCALE_PROFILE | mix1-decode-stdio | dix1-encode PROFILE INPUT META_JSON OUTPUT | dix1-decode INPUT OUTPUT | dix1-encode-stdio PROFILE META_JSON | dix1-decode-stdio | dix2-encode-stdio PROFILE META_JSON | dix2-decode-stdio | learned-encode MODE MODEL INPUT META_JSON OUTPUT | learned-decode MODEL INPUT OUTPUT"
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
    use super::GovernedRawBoundary;
    use std::fs;

    #[test]
    #[cfg(target_os = "linux")]
    fn rejects_lexical_descendants_and_canonical_directory_aliases() {
        use std::os::unix::fs::symlink;

        let base =
            std::env::temp_dir().join(format!("optimum_v2_bgf1_path_guard_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let governed_root = base.join("governed/raw");
        fs::create_dir_all(&governed_root).unwrap();
        let governed_input = governed_root.join("input.lqraw");
        fs::write(&governed_input, b"synthetic").unwrap();
        let boundary = GovernedRawBoundary::new(&governed_root).unwrap();

        assert!(boundary
            .existing_path(&governed_root.join("missing.lqraw"), "INPUT")
            .is_err());

        let alias = base.join("alias");
        symlink(&governed_root, &alias).unwrap();
        assert!(boundary
            .existing_path(&alias.join("input.lqraw"), "INPUT")
            .is_err());
        assert!(boundary.output_path(&alias.join("missing.lmo")).is_err());

        let hardlink = base.join("hardlink.lqraw");
        fs::hard_link(&governed_input, &hardlink).unwrap();
        let hardlink = boundary.existing_path(&hardlink, "INPUT").unwrap();
        let error = boundary.open_existing(&hardlink, "INPUT").unwrap_err();
        assert!(error.contains("hard-linked"));
        let hardlink_output = base.join("hardlink-output.lmo");
        fs::hard_link(&governed_input, &hardlink_output).unwrap();
        let hardlink_output = boundary.output_path(&hardlink_output).unwrap();
        let error = boundary.open_output(&hardlink_output).unwrap_err();
        assert!(error.contains("refuse existing"));
        assert_eq!(fs::read(&governed_input).unwrap(), b"synthetic");

        let scratch = base.join("scratch.lqraw");
        fs::write(&scratch, b"synthetic").unwrap();
        assert_eq!(
            boundary.existing_path(&scratch, "INPUT").unwrap(),
            fs::canonicalize(&scratch).unwrap()
        );
        assert_eq!(
            boundary
                .output_path(&base.join("scratch.lmo"))
                .unwrap()
                .path(),
            fs::canonicalize(&base).unwrap().join("scratch.lmo")
        );

        let approved_parent = base.join("approved-parent");
        let substitute_parent = base.join("substitute-parent");
        fs::create_dir_all(&approved_parent).unwrap();
        fs::create_dir_all(&substitute_parent).unwrap();
        fs::write(approved_parent.join("input.lqraw"), b"approved").unwrap();
        fs::write(substitute_parent.join("input.lqraw"), b"substitute").unwrap();
        let approved_input = boundary
            .existing_path(&approved_parent.join("input.lqraw"), "INPUT")
            .unwrap();
        fs::rename(&approved_parent, base.join("approved-parent-moved")).unwrap();
        symlink(&substitute_parent, &approved_parent).unwrap();
        let error = boundary
            .open_existing(&approved_input, "INPUT")
            .unwrap_err();
        assert!(error.contains("changed between validation and open"));

        let output_parent = base.join("output-parent");
        let substitute_output_parent = base.join("substitute-output-parent");
        fs::create_dir_all(&output_parent).unwrap();
        fs::create_dir_all(&substitute_output_parent).unwrap();
        let approved_output = boundary
            .output_path(&output_parent.join("result.lmo"))
            .unwrap();
        fs::rename(&output_parent, base.join("output-parent-moved")).unwrap();
        fs::rename(&substitute_output_parent, &output_parent).unwrap();
        let error = boundary.open_output(&approved_output).unwrap_err();
        assert!(error.contains("OUTPUT parent changed before creation"));
        assert_eq!(
            fs::read_dir(&output_parent).unwrap().count(),
            0,
            "a same-path replacement parent must remain untouched",
        );
        let _ = fs::remove_dir_all(base);
    }
}
