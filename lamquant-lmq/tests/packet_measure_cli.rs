#![cfg(feature = "python")]

use std::io::Write as _;
use std::process::{Command, Stdio};

#[test]
fn packet_measure_cli_emits_frozen_lmqp1_bytes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lmq-packet-measure"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn packet measurement CLI");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(
            br#"{"schema":"lamquant.lmq-packet-measure-request/v1","windows":[{"tokens":[0,1,2,1],"schedule":[3,3],"alphabet":3,"n_channels":2,"n_samples":2,"backend_meta":[]}]}"#,
        )
        .expect("write request");
    let output = child
        .wait_with_output()
        .expect("collect packet measurement");

    assert!(
        output.status.success(),
        "packet measurement failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 response"),
        concat!(
            "{\"schema\":\"lamquant.lmq-packet-measure-response/v1\",",
            "\"windows\":[{\"packet_bytes\":48,",
            "\"packet_sha256\":",
            "\"3825c8fcd3cd303723d150e07044e4eadda21bbe119a1dbf9c53635b89fea9d2\"}]}\n"
        )
    );
}
