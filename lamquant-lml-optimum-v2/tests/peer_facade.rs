use lamquant_lml_optimum_v2::mix1::Mix1Codec;
use lamquant_lml_optimum_v2::{
    peer_implementation_identity, PeerCodec, PeerEncodeContext, PeerPacketKind, PEER_KERNEL_ID,
    PEER_MAX_CHANNELS, PEER_MAX_PACKET_BYTES, PEER_MAX_PEAK_BYTES, PEER_MAX_SAMPLES,
    PEER_MAX_SCRATCH_BYTES, PEER_MAX_SIGNAL_BYTES, PEER_MAX_VALUES,
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn fixture() -> Vec<Vec<i64>> {
    vec![
        (0..128)
            .map(|time| i64::from((time * 7 + time / 11) % 257) - 128)
            .collect(),
        (0..128)
            .map(|time| i64::from((time * 5 + time / 13) % 193) - 96)
            .collect(),
    ]
}

#[test]
fn production_facade_is_byte_equal_to_frozen_peer_portfolio() {
    let signal = fixture();
    let context = PeerEncodeContext {
        sample_rate_mhz: 256_000,
        bit_depth: 16,
    };

    let encoded = PeerCodec
        .encode_window_with_report(&signal, context)
        .expect("encode through production facade");
    let direct = Mix1Codec
        .encode_best_peer_window(&signal, context.sample_rate_mhz, context.bit_depth)
        .expect("encode through frozen peer portfolio");

    assert_eq!(encoded.packet, direct);
    let packet_digest = Sha256::digest(&encoded.packet)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        packet_digest,
        "edd9427620badeee8b239a90f4800fb772a4af3826499605284eb07b04349377"
    );
    assert_eq!(encoded.report.packet_bytes, encoded.packet.len());
    assert_eq!(encoded.report.source_value_count, 256);
    assert_eq!(encoded.report.channel_count, 2);
    assert_eq!(encoded.report.sample_count, 128);
    assert!(matches!(
        encoded.report.selected_kind,
        PeerPacketKind::Mix1
            | PeerPacketKind::Multivariate
            | PeerPacketKind::HierarchicalMultivariate
            | PeerPacketKind::ChannelContext
            | PeerPacketKind::CommonMode
            | PeerPacketKind::PermutedCommonMode
            | PeerPacketKind::AdaptivePermuted
            | PeerPacketKind::CompactCommon
            | PeerPacketKind::Alias
            | PeerPacketKind::Bitplane
    ));

    let decoded = PeerCodec
        .decode_window(&encoded.packet)
        .expect("decode production packet");
    assert_eq!(decoded.samples, signal);
    assert_eq!(decoded.context, context);
}

#[test]
fn production_contract_exposes_exact_identity_and_resource_limits() {
    const {
        assert!(PEER_MAX_CHANNELS == 256);
        assert!(PEER_MAX_SAMPLES == 32_768);
        assert!(PEER_MAX_VALUES == 131_072);
        assert!(PEER_MAX_SIGNAL_BYTES == 1024 * 1024);
        assert!(PEER_MAX_PACKET_BYTES == 64 * 1024 * 1024);
        assert!(PEER_MAX_SCRATCH_BYTES >= 10 * 1024 * 1024 * 1024);
        assert!(
            PEER_MAX_PEAK_BYTES
                == PEER_MAX_SCRATCH_BYTES + PEER_MAX_SIGNAL_BYTES + 2 * PEER_MAX_PACKET_BYTES
        );
    }
    assert_eq!(
        PEER_KERNEL_ID,
        "org.quitetall.lamquant.lml-optimum-v2.peer-v4"
    );
    assert_eq!(
        peer_implementation_identity().feature_set,
        if cfg!(feature = "parallel") {
            "parallel"
        } else {
            "none"
        }
    );

    let too_many_channels = vec![vec![0_i64]; PEER_MAX_CHANNELS + 1];
    assert!(PeerCodec
        .encode_window(
            &too_many_channels,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .is_err());

    let too_many_samples = vec![vec![0_i64; PEER_MAX_SAMPLES + 1]];
    assert!(PeerCodec
        .encode_window(
            &too_many_samples,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .is_err());

    let too_many_values = vec![vec![0_i64; PEER_MAX_VALUES / 2 + 1]; 2];
    assert!(PeerCodec
        .encode_window(
            &too_many_values,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .is_err());

    let oversized_packet = vec![0_u8; PEER_MAX_PACKET_BYTES as usize + 1];
    assert!(PeerCodec.decode_window(&oversized_packet).is_err());
    assert!(PeerCodec.inspect_packet(&oversized_packet).is_err());
}

#[test]
fn independent_python_decoder_closes_frozen_facade_packet() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(meta_root) = manifest.parent().and_then(Path::parent) else {
        panic!("Optimum-v2 manifest has no meta-repository ancestor");
    };
    let decoder = meta_root.join("tools/hbwc_bench/optimum_v2_mix1_codec.py");
    let cookbook = meta_root.join("training/cookbooks/lamquant/python");
    if !decoder.is_file() || !cookbook.is_dir() {
        assert_ne!(
            std::env::var("LAMQUANT_REQUIRE_PEER_PYTHON_ORACLE").as_deref(),
            Ok("1"),
            "required independent Python peer decoder is unavailable"
        );
        eprintln!("skipping meta-repository Python oracle outside LamQuant checkout");
        return;
    }

    let signal = fixture();
    let context = PeerEncodeContext {
        sample_rate_mhz: 256_000,
        bit_depth: 16,
    };
    let packet = PeerCodec
        .encode_window(&signal, context)
        .expect("encode frozen facade vector");
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let mut child = Command::new(&python)
        .arg(&decoder)
        .arg("decode-stdio")
        .env("PYTHONPATH", &cookbook)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("cannot start {python}: {error}"));
    child
        .stdin
        .take()
        .expect("Python decoder stdin")
        .write_all(&packet)
        .expect("write peer packet to Python decoder");
    let output = child.wait_with_output().expect("wait for Python decoder");
    assert!(
        output.status.success(),
        "Python decoder failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut expected = Vec::new();
    expected.extend_from_slice(b"LQR1");
    expected.extend_from_slice(&[1, 4, context.bit_depth, 0]);
    expected.extend_from_slice(&context.sample_rate_mhz.to_le_bytes());
    expected.extend_from_slice(&(signal.len() as u32).to_le_bytes());
    expected.extend_from_slice(&(signal[0].len() as u32).to_le_bytes());
    for channel in &signal {
        for &sample in channel {
            expected.extend_from_slice(&(sample as i32).to_le_bytes());
        }
    }
    assert_eq!(output.stdout, expected);
}

#[test]
fn report_rejects_noncanonical_or_unknown_packets() {
    assert!(PeerCodec.inspect_packet(b"").is_err());

    let signal = fixture();
    let packet = PeerCodec
        .encode_window(
            &signal,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .expect("encode packet");
    let mut corrupted = packet;
    corrupted[0] ^= 1;
    assert!(PeerCodec.inspect_packet(&corrupted).is_err());
}
