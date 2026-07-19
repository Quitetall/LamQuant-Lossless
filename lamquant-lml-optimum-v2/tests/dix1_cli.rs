use std::fs;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_os = "linux")]
fn make_fifo(path: &std::path::Path) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
}

#[cfg(target_os = "linux")]
fn assert_rejected_promptly(mut command: Command) {
    use std::io::Read;
    use std::process::Stdio;

    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    for _ in 0..100 {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(!status.success());
            let mut stderr = String::new();
            child
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut stderr)
                .unwrap();
            assert!(
                stderr.contains("is not a regular file"),
                "unexpected special-file diagnostic: {stderr}"
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    child.kill().unwrap();
    let _ = child.wait();
    panic!("construction worker blocked on a special-file operand");
}

fn lqraw_fixture(channels: usize, samples: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, 16, 0]);
    bytes.extend_from_slice(&250_000u32.to_le_bytes());
    bytes.extend_from_slice(&(channels as u32).to_le_bytes());
    bytes.extend_from_slice(&(samples as u32).to_le_bytes());
    for channel in 0..channels {
        for sample in 0..samples {
            let base = i32::try_from(sample).unwrap() * 7 - 400;
            let value = if channel == 0 {
                base
            } else {
                base + i32::try_from(sample % 9).unwrap() - 4
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn stdio_worker(binary: &str, arguments: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(binary)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn dix1_stdio_worker_round_trips_every_profile_without_scratch_files() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture(2, 257);
    let metadata = r#"{"channel_labels":["EEG Fp1-Ref","EEG Fp2-Ref"]}"#;

    for (profile, wire_profile) in [
        ("product", 0u8),
        ("native", 1),
        ("raw", 2),
        ("delta", 3),
        ("incidence", 4),
        ("no-incidence", 5),
    ] {
        let encoded = stdio_worker(binary, &["dix1-encode-stdio", profile, metadata], &raw);
        assert!(
            encoded.status.success(),
            "stdio encode failed: {}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        assert_eq!(&encoded.stdout[..12], b"LMO1\x03\x00\x03DIX1\x02");
        assert_eq!(encoded.stdout[14], wire_profile);

        let decoded = stdio_worker(binary, &["dix1-decode-stdio"], &encoded.stdout);
        assert!(
            decoded.status.success(),
            "stdio decode failed: {}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert_eq!(decoded.stdout, raw);
    }
}

#[test]
#[cfg(target_os = "linux")]
fn dix1_benchmark_worker_round_trips_every_profile_fail_closed() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "optimum_v2_dix1_cli_{}_{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let metadata = root.join("metadata.json");
    let raw_bytes = lqraw_fixture(2, 257);
    fs::write(&raw, &raw_bytes).unwrap();
    fs::write(
        &metadata,
        br#"{"channel_labels":["EEG Fp1-Ref","EEG Fp2-Ref"]}"#,
    )
    .unwrap();

    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    for (profile, wire_profile) in [
        ("product", 0u8),
        ("native", 1),
        ("raw", 2),
        ("delta", 3),
        ("incidence", 4),
        ("no-incidence", 5),
    ] {
        let packet = root.join(format!("{profile}.dix1"));
        let reconstructed = root.join(format!("{profile}.lqraw"));
        assert!(Command::new(binary)
            .args([
                "dix1-encode",
                profile,
                raw.to_str().unwrap(),
                metadata.to_str().unwrap(),
                packet.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success());
        let packet_bytes = fs::read(&packet).unwrap();
        assert_eq!(&packet_bytes[..12], b"LMO1\x03\x00\x03DIX1\x02");
        assert_eq!(packet_bytes[14], wire_profile);
        assert!(Command::new(binary)
            .args([
                "dix1-decode",
                packet.to_str().unwrap(),
                reconstructed.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success());
        assert_eq!(fs::read(&reconstructed).unwrap(), raw_bytes);

        let occupied_encode = Command::new(binary)
            .args([
                "dix1-encode",
                profile,
                raw.to_str().unwrap(),
                metadata.to_str().unwrap(),
                packet.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!occupied_encode.status.success());
        assert!(String::from_utf8_lossy(&occupied_encode.stderr)
            .contains("refuse existing construction worker OUTPUT"));
        let occupied_decode = Command::new(binary)
            .args([
                "dix1-decode",
                packet.to_str().unwrap(),
                reconstructed.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!occupied_decode.status.success());
        assert!(String::from_utf8_lossy(&occupied_decode.stderr)
            .contains("refuse existing construction worker OUTPUT"));
    }

    let descriptor = root.join("descriptor.json");
    assert!(Command::new(binary)
        .args(["describe", descriptor.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let value: serde_json::Value = serde_json::from_slice(&fs::read(descriptor).unwrap()).unwrap();
    assert_eq!(
        value["codec"],
        "LamQuant Optimum v2 native, MIX1, DIX1/DIX2 construction, and BGF1 learned carrier"
    );
    assert_eq!(
        value["wire"],
        "LMO1-v3/BGF1-v1/OV2P-v2-MIX1/DIX1-v2/DIX2-v1-construction"
    );
    assert_eq!(
        value["dix1_worker"]["encode"],
        "dix1-encode PROFILE INPUT META_JSON OUTPUT"
    );
    assert_eq!(value["dix1_worker"]["decode"], "dix1-decode INPUT OUTPUT");
    assert_eq!(
        value["dix1_worker"]["encode_stdio"],
        "dix1-encode-stdio PROFILE META_JSON"
    );
    assert_eq!(value["dix1_worker"]["decode_stdio"], "dix1-decode-stdio");
    assert_eq!(value["dix1_worker"]["body_version"], 2);
    assert_eq!(value["dix1_worker"]["construction_private"], true);
    assert_eq!(
        value["dix2_worker"]["encode_stdio"],
        "dix2-encode-stdio PROFILE META_JSON"
    );
    assert_eq!(value["dix2_worker"]["decode_stdio"], "dix2-decode-stdio");
    assert_eq!(value["dix2_worker"]["body_version"], 1);
    assert_eq!(value["dix2_worker"]["construction_private"], true);
    assert_eq!(
        value["dix2_worker"]["profiles"],
        serde_json::json!(["product", "native", "raw", "delta", "temporal", "tree"])
    );
    assert_eq!(
        value["dix1_worker"]["profiles"],
        serde_json::json!([
            "product",
            "native",
            "raw",
            "delta",
            "incidence",
            "no-incidence"
        ])
    );

    let governed = "/mnt/4tb/LamQuant/outputs/optimum-v2-development-v2-2k/raw/missing.lqraw";
    let must_not_exist = root.join("governed-source.dix1");
    let rejected = Command::new(binary)
        .args([
            "dix1-encode",
            "product",
            governed,
            metadata.to_str().unwrap(),
            must_not_exist.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr)
        .contains("within the governed construction raw root"));
    assert!(!must_not_exist.exists());

    let fifo_input = root.join("input.fifo");
    let fifo_metadata = root.join("metadata.fifo");
    let fifo_packet = root.join("packet.fifo");
    make_fifo(&fifo_input);
    make_fifo(&fifo_metadata);
    make_fifo(&fifo_packet);
    let fifo_input_output = root.join("fifo-input.dix1");
    let mut fifo_input_command = Command::new(binary);
    fifo_input_command.args([
        "dix1-encode",
        "product",
        fifo_input.to_str().unwrap(),
        metadata.to_str().unwrap(),
        fifo_input_output.to_str().unwrap(),
    ]);
    assert_rejected_promptly(fifo_input_command);
    assert!(!fifo_input_output.exists());

    let fifo_metadata_output = root.join("fifo-metadata.dix1");
    let mut fifo_metadata_command = Command::new(binary);
    fifo_metadata_command.args([
        "dix1-encode",
        "product",
        raw.to_str().unwrap(),
        fifo_metadata.to_str().unwrap(),
        fifo_metadata_output.to_str().unwrap(),
    ]);
    assert_rejected_promptly(fifo_metadata_command);
    assert!(!fifo_metadata_output.exists());

    let fifo_decode_output = root.join("fifo-reconstruction.lqraw");
    let mut fifo_decode_command = Command::new(binary);
    fifo_decode_command.args([
        "dix1-decode",
        fifo_packet.to_str().unwrap(),
        fifo_decode_output.to_str().unwrap(),
    ]);
    assert_rejected_promptly(fifo_decode_command);
    assert!(!fifo_decode_output.exists());

    let device_output = root.join("device-source.dix1");
    let mut device_command = Command::new(binary);
    device_command.args([
        "dix1-encode",
        "product",
        "/dev/null",
        metadata.to_str().unwrap(),
        device_output.to_str().unwrap(),
    ]);
    assert_rejected_promptly(device_command);
    assert!(!device_output.exists());

    let _ = fs::remove_dir_all(root);
}
