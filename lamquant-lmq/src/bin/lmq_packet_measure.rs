use std::io::{self, Read as _, Write as _};
use std::process::ExitCode;

use lamquant_lmq::backend::NeuralTokens;
use lamquant_lmq::shell::encode_token_packet;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const INPUT_LIMIT: u64 = 64 * 1024 * 1024;
const WINDOW_LIMIT: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    windows: Vec<TokenWindow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenWindow {
    tokens: Vec<i32>,
    schedule: Vec<u8>,
    alphabet: u16,
    n_channels: u16,
    n_samples: u32,
    backend_meta: Vec<u8>,
}

#[derive(Serialize)]
struct Response {
    schema: &'static str,
    windows: Vec<PacketMeasurement>,
}

#[derive(Serialize)]
struct PacketMeasurement {
    packet_bytes: usize,
    packet_sha256: String,
}

fn measure() -> Result<Response, String> {
    let mut input = Vec::new();
    io::stdin()
        .take(INPUT_LIMIT + 1)
        .read_to_end(&mut input)
        .map_err(|error| format!("could not read request: {error}"))?;
    if input.is_empty() || input.len() as u64 > INPUT_LIMIT {
        return Err("request size is outside allowed range".into());
    }
    let request: Request =
        serde_json::from_slice(&input).map_err(|error| format!("invalid request JSON: {error}"))?;
    if request.schema != "lamquant.lmq-packet-measure-request/v1" {
        return Err("unsupported request schema".into());
    }
    if request.windows.is_empty() || request.windows.len() > WINDOW_LIMIT {
        return Err("window count is outside allowed range".into());
    }
    let mut measurements = Vec::with_capacity(request.windows.len());
    for window in request.windows {
        let packet = encode_token_packet(&NeuralTokens {
            tokens: window.tokens,
            schedule: window.schedule,
            alphabet: window.alphabet,
            n_channels: window.n_channels,
            n_samples: window.n_samples,
            backend_meta: window.backend_meta,
        })
        .map_err(|error| format!("LMQP1 encoding failed: {error}"))?;
        measurements.push(PacketMeasurement {
            packet_bytes: packet.len(),
            packet_sha256: format!("{:x}", Sha256::digest(&packet)),
        });
    }
    Ok(Response {
        schema: "lamquant.lmq-packet-measure-response/v1",
        windows: measurements,
    })
}

fn main() -> ExitCode {
    match measure() {
        Ok(response) => {
            let stdout = io::stdout();
            let mut output = stdout.lock();
            if let Err(error) = serde_json::to_writer(&mut output, &response) {
                eprintln!("could not write response: {error}");
                return ExitCode::from(2);
            }
            if let Err(error) = output.write_all(b"\n") {
                eprintln!("could not finish response: {error}");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
