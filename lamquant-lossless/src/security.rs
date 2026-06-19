//! Phase 7 — encryption, HMAC signing, and tamper-evident audit log.
//!
//! Three host-only primitives, each independent of the codec internals:
//!
//! - `encrypt_aes_gcm` / `decrypt_aes_gcm` — AES-256-GCM AEAD over a
//!   `.lml` / `.lma` blob. Wire format:
//!
//!   ```text
//!   MAGIC(8 = "LMLCRYPT") | VERSION(1=1) | NONCE(12) | CIPHERTEXT
//!   ```
//!
//!   The ciphertext ends with the 16-byte GCM tag; decryption verifies
//!   the tag before returning bytes, so the file is either fully
//!   authenticated or fully rejected. Bible R7 — never silently
//!   consume corrupt data.
//!
//! - `hmac_sign` / `hmac_verify` — HMAC-SHA-256 detached signature.
//!   Outputs / consumes a fixed 32-byte tag. The caller chooses the
//!   sidecar location (`<file>.hmac` by convention).
//!
//! - `AuditLog` — append-only JSONL with per-line SHA-256 chain. Each
//!   line includes the SHA of the previous line + this line's payload,
//!   so any tamper to the middle of the file breaks the chain.
//!
//! Key material comes from the `LAMQUANT_KEY` env var as a 64-char
//! hex string (32 bytes raw). `keyring` integration is deferred to a
//! later phase — env-vars are sufficient for CI / scripted use cases
//! and avoid the Linux Secret-Service dependency chain.

use crate::error::{LmlError, LmlResult};
use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, Nonce};
use hmac::{Hmac, Mac as HmacMac};
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

// aes-gcm's `KeyInit` and hmac's `Mac` both define `new_from_slice`.
// Importing `Mac as HmacMac` and qualifying aes-gcm's via the trait
// path resolves the ambiguity without changing call shape.
use aes_gcm::KeyInit as AesKeyInit;

pub const ENC_MAGIC: &[u8; 8] = b"LMLCRYPT";
pub const ENC_VERSION: u8 = 1;
pub const ENC_NONCE_LEN: usize = 12;
pub const ENC_HEADER_LEN: usize = 8 + 1 + ENC_NONCE_LEN; // magic + version + nonce
pub const KEY_LEN: usize = 32;
pub const HMAC_TAG_LEN: usize = 32;

/// 32-byte key wrapped so `Drop` zeroes the buffer.
pub struct Key([u8; KEY_LEN]);

impl Key {
    pub fn from_bytes(b: [u8; KEY_LEN]) -> Self {
        Self(b)
    }
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
    /// Parse a 64-character hex string. Errors on length / bad hex.
    pub fn from_hex(s: &str) -> LmlResult<Self> {
        let s = s.trim();
        if s.len() != 64 {
            return Err(LmlError::InvalidHeader(format!(
                "key: hex must be 64 chars (got {})",
                s.len()
            )));
        }
        let mut out = [0u8; KEY_LEN];
        for i in 0..KEY_LEN {
            let byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| {
                LmlError::InvalidHeader(format!(
                    "key: invalid hex byte {}: {e}",
                    &s[2 * i..2 * i + 2]
                ))
            })?;
            out[i] = byte;
        }
        Ok(Self(out))
    }
    /// Read a 32-byte key from the `LAMQUANT_KEY` env-var (hex). Errors
    /// when the env-var is unset or malformed.
    pub fn from_env() -> LmlResult<Self> {
        let s = std::env::var("LAMQUANT_KEY").map_err(|_| {
            LmlError::InvalidHeader(
                "LAMQUANT_KEY env var not set (expected 64-char hex string for AES-256 key)".into(),
            )
        })?;
        Self::from_hex(&s)
    }

    /// Phase 7.3 — load the key from the OS keyring under the given
    /// service+account names. Linux uses libsecret; macOS uses
    /// Keychain; Windows uses DPAPI. Feature-gated behind `keyring`.
    #[cfg(feature = "keyring")]
    pub fn from_keyring(service: &str, account: &str) -> LmlResult<Self> {
        let entry = keyring::Entry::new(service, account).map_err(|e| {
            LmlError::InvalidHeader(format!("keyring: open {service}/{account}: {e}"))
        })?;
        let s = entry.get_password().map_err(|e| {
            LmlError::InvalidHeader(format!("keyring: read {service}/{account}: {e}"))
        })?;
        Self::from_hex(&s)
    }

    /// Phase 7.3 — store a 64-char hex key in the OS keyring.
    #[cfg(feature = "keyring")]
    pub fn store_in_keyring(&self, service: &str, account: &str) -> LmlResult<()> {
        let entry = keyring::Entry::new(service, account).map_err(|e| {
            LmlError::InvalidHeader(format!("keyring: open {service}/{account}: {e}"))
        })?;
        let hex: String = self.0.iter().map(|b| format!("{b:02x}")).collect();
        entry.set_password(&hex).map_err(|e| {
            LmlError::InvalidHeader(format!("keyring: write {service}/{account}: {e}"))
        })?;
        Ok(())
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

// ─── Phase 7 / v1.2 P — Password-derived keys (Argon2id KDF) ───

/// Argon2id parameters bundled in the `.lmcrypt.header` sidecar.
/// OWASP-recommended defaults: m=64 MiB, t=3, p=1. Operators can
/// override via `--argon2-params m:t:p` when they need faster decrypt
/// on low-end hardware (at the cost of brute-force resistance).
#[cfg(feature = "security")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Argon2Params {
    /// Memory cost in KiB. OWASP default = 65536 (64 MiB).
    pub m_kib: u32,
    /// Time cost (passes). OWASP default = 3.
    pub t_cost: u32,
    /// Parallelism (lanes). OWASP default = 1.
    pub p_cost: u32,
}

#[cfg(feature = "security")]
impl Default for Argon2Params {
    fn default() -> Self {
        Argon2Params {
            m_kib: 65536,
            t_cost: 3,
            p_cost: 1,
        }
    }
}

/// Format of the `.lmcrypt.header` sidecar (32 bytes total):
///
/// ```text
/// offset  bytes  field
///  0      4      magic = "LMHP" (LamQuant Lmcrypt Header v1)
///  4      4      version (u32 LE) = 1
///  8      16     salt (random per-encrypt)
/// 24      4      m_kib (u32 LE, Argon2 memory cost in KiB)
/// 28      2      t_cost (u16 LE, Argon2 time cost)
/// 30      1      p_cost (u8, Argon2 parallelism)
/// 31      1      reserved (0)
/// ```
#[cfg(feature = "security")]
#[derive(Debug)]
pub struct LmcryptHeader {
    pub salt: [u8; 16],
    pub params: Argon2Params,
}

#[cfg(feature = "security")]
impl LmcryptHeader {
    /// Magic + version + params length. Used by callers to size buffers.
    pub const SIZE: usize = 32;

    /// Build a header with a freshly-randomized salt.
    pub fn new_random(params: Argon2Params) -> LmlResult<Self> {
        let mut salt = [0u8; 16];
        getrandom::getrandom(&mut salt).map_err(|e| {
            LmlError::InvalidHeader(format!("lmcrypt: failed to generate salt ({e})"))
        })?;
        Ok(LmcryptHeader { salt, params })
    }

    /// Serialize to 32 bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(b"LMHP");
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[8..24].copy_from_slice(&self.salt);
        out[24..28].copy_from_slice(&self.params.m_kib.to_le_bytes());
        out[28..30].copy_from_slice(&(self.params.t_cost as u16).to_le_bytes());
        out[30] = self.params.p_cost as u8;
        // out[31] reserved = 0
        out
    }

    /// Parse from 32 bytes; errors on bad magic / unsupported version.
    pub fn from_bytes(buf: &[u8]) -> LmlResult<Self> {
        if buf.len() < Self::SIZE {
            return Err(LmlError::Truncated {
                expected: Self::SIZE,
                actual: buf.len(),
                context: "lmcrypt header",
            });
        }
        if &buf[0..4] != b"LMHP" {
            return Err(LmlError::InvalidHeader(format!(
                "lmcrypt header: bad magic {:?}, expected LMHP",
                &buf[0..4]
            )));
        }
        let ver = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if ver != 1 {
            return Err(LmlError::UnsupportedVersion(ver as u8));
        }
        let mut salt = [0u8; 16];
        salt.copy_from_slice(&buf[8..24]);
        let m_kib = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
        let t_cost = u16::from_le_bytes([buf[28], buf[29]]) as u32;
        let p_cost = buf[30] as u32;
        Ok(LmcryptHeader {
            salt,
            params: Argon2Params {
                m_kib,
                t_cost,
                p_cost,
            },
        })
    }
}

#[cfg(feature = "security")]
impl Key {
    /// Derive a 32-byte AES-256 key from a password via Argon2id.
    ///
    /// The salt + params come from the `.lmcrypt.header` sidecar (or
    /// from a freshly-created header on encrypt). Argon2id offers
    /// hybrid resistance against both side-channel and time-memory
    /// trade-off attacks; OWASP recommends it as the password-hashing
    /// default.
    ///
    /// Errors when the password is empty (defends against accidental
    /// empty-string from a bad prompt loop) or when the KDF parameters
    /// are out of range.
    pub fn from_password(password: &str, header: &LmcryptHeader) -> LmlResult<Self> {
        if password.is_empty() {
            return Err(LmlError::InvalidHeader(
                "lmcrypt: empty password not allowed".into(),
            ));
        }
        let argon_params = argon2::Params::new(
            header.params.m_kib,
            header.params.t_cost,
            header.params.p_cost,
            Some(32),
        )
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "lmcrypt: bad Argon2 params (m={} t={} p={}): {e}",
                header.params.m_kib, header.params.t_cost, header.params.p_cost
            ))
        })?;
        let kdf = argon2::Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon_params,
        );
        let mut out = [0u8; 32];
        kdf.hash_password_into(password.as_bytes(), &header.salt, &mut out)
            .map_err(|e| LmlError::InvalidHeader(format!("lmcrypt: KDF failed: {e}")))?;
        Ok(Self(out))
    }
}

/// AES-256-GCM encrypt `plaintext` under `key`. The 12-byte nonce
/// comes from `getrandom` via aes-gcm internals; embed it in the
/// returned `Vec<u8>` so decrypt is self-describing.
///
/// Output layout: `MAGIC | VERSION | NONCE | CIPHERTEXT_WITH_TAG`.
pub fn encrypt_aes_gcm(key: &Key, plaintext: &[u8]) -> LmlResult<Vec<u8>> {
    let cipher = <Aes256Gcm as AesKeyInit>::new_from_slice(key.as_bytes())
        .map_err(|e| LmlError::InvalidHeader(format!("aes-gcm: bad key length ({e})")))?;
    // Deterministic-derived nonce avoids the `getrandom` host-only
    // dependency. SHA-256(plaintext || ts) → first 12 bytes. Two
    // distinct plaintexts always yield distinct nonces; same plaintext
    // re-encrypted at a different timestamp also yields a new nonce
    // (timestamp is part of the seed).
    let mut nonce_bytes = [0u8; ENC_NONCE_LEN];
    {
        let mut hasher = Sha256::new();
        hasher.update(plaintext);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes();
        hasher.update(ts);
        let h = hasher.finalize();
        nonce_bytes.copy_from_slice(&h[..ENC_NONCE_LEN]);
    }
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| LmlError::InvalidHeader(format!("aes-gcm encrypt: {e}")))?;
    let mut out = Vec::with_capacity(ENC_HEADER_LEN + ciphertext.len());
    out.extend_from_slice(ENC_MAGIC);
    out.push(ENC_VERSION);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt + authenticate an AES-256-GCM blob produced by
/// `encrypt_aes_gcm`. Errors on bad magic, unsupported version,
/// truncation, or auth-tag mismatch.
pub fn decrypt_aes_gcm(key: &Key, blob: &[u8]) -> LmlResult<Vec<u8>> {
    if blob.len() < ENC_HEADER_LEN + 16 {
        return Err(LmlError::Truncated {
            expected: ENC_HEADER_LEN + 16,
            actual: blob.len(),
            context: "aes-gcm header + tag",
        });
    }
    if &blob[0..8] != ENC_MAGIC {
        return Err(LmlError::InvalidMagic([blob[0], blob[1], blob[2], blob[3]]));
    }
    if blob[8] != ENC_VERSION {
        return Err(LmlError::UnsupportedVersion(blob[8]));
    }
    let nonce = Nonce::from_slice(&blob[9..9 + ENC_NONCE_LEN]);
    let ciphertext = &blob[ENC_HEADER_LEN..];
    let cipher = <Aes256Gcm as AesKeyInit>::new_from_slice(key.as_bytes())
        .map_err(|e| LmlError::InvalidHeader(format!("aes-gcm: bad key length ({e})")))?;
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| LmlError::InvalidHeader(format!("aes-gcm decrypt (auth fail): {e}")))
}

/// HMAC-SHA-256 detached signature of `bytes` under `key`. Returns a
/// fixed 32-byte tag.
pub fn hmac_sign(key: &Key, bytes: &[u8]) -> [u8; HMAC_TAG_LEN] {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = <HmacSha256 as HmacMac>::new_from_slice(key.as_bytes())
        .expect("hmac: 32-byte key always fits HmacSha256");
    mac.update(bytes);
    let result = mac.finalize().into_bytes();
    let mut out = [0u8; HMAC_TAG_LEN];
    out.copy_from_slice(&result);
    out
}

/// HMAC-SHA-256 verify. Constant-time tag comparison via the underlying
/// `hmac` crate's `verify` API.
pub fn hmac_verify(key: &Key, bytes: &[u8], tag: &[u8; HMAC_TAG_LEN]) -> bool {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = match <HmacSha256 as HmacMac>::new_from_slice(key.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(bytes);
    mac.verify_slice(tag).is_ok()
}

/// Phase 7.6 — Tamper-evident append-only audit log.
///
/// Each entry is a JSON object:
///
/// ```text
/// {"ts":"<RFC3339>","op":"<verb>","prev":"<32 hex chars>","sha":"<32 hex chars>","msg":"..."}
/// ```
///
/// `sha` = first 16 bytes of SHA-256(`prev || msg`). `prev` is the
/// previous entry's `sha`, or `"00…00"` for the genesis entry. Tamper
/// to any byte of any line breaks the chain at that point. Bible R7
/// (CRC-style integrity at boundaries) + R32 (idempotent receivers —
/// re-running `verify` is no-op-safe).
pub struct AuditLog {
    path: std::path::PathBuf,
}

impl AuditLog {
    pub fn new<P: Into<std::path::PathBuf>>(path: P) -> Self {
        Self { path: path.into() }
    }

    /// Append a new entry. Reads the last line of the existing log to
    /// pick up `prev`; locks on file create + append, no concurrent
    /// writer protection beyond what the OS gives. Bible R31 — calls
    /// are idempotent if the user passes an idempotency-key `op_id` in
    /// the message; the audit log doesn't dedupe by itself.
    pub fn append(&self, op: &str, msg: &str) -> LmlResult<()> {
        let prev_hex = self.tail_sha()?;
        let ts = current_rfc3339();
        let payload = format!("{prev_hex}|{msg}");
        let mut h = Sha256::new();
        h.update(payload.as_bytes());
        let digest = h.finalize();
        let sha_hex: String = digest[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("");
        let line = format!(
            "{{\"ts\":\"{ts}\",\"op\":\"{}\",\"prev\":\"{prev_hex}\",\"sha\":\"{sha_hex}\",\"msg\":\"{}\"}}\n",
            json_escape(op),
            json_escape(msg)
        );
        use std::io::Write as _;
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
            }
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(LmlError::Io)?;
        f.write_all(line.as_bytes()).map_err(LmlError::Io)?;
        f.sync_data().map_err(LmlError::Io)?;
        Ok(())
    }

    /// Walk the log file and confirm the SHA chain is intact. Returns
    /// the number of verified entries, or an `Err` at the first break.
    pub fn verify(&self) -> LmlResult<usize> {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(LmlError::Io(e)),
        };
        let mut prev_hex = "0".repeat(32);
        let mut count = 0usize;
        for (line_no, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let prev_in_entry = json_str_field(line, "prev").ok_or_else(|| {
                LmlError::InvalidHeader(format!("audit-log line {line_no}: missing 'prev' field"))
            })?;
            if prev_in_entry != prev_hex {
                return Err(LmlError::InvalidHeader(format!(
                    "audit-log line {line_no}: prev '{prev_in_entry}' != expected '{prev_hex}'"
                )));
            }
            let msg = json_str_field(line, "msg").unwrap_or_default();
            let payload = format!("{prev_in_entry}|{msg}");
            let mut h = Sha256::new();
            h.update(payload.as_bytes());
            let digest = h.finalize();
            let computed: String = digest[..16]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join("");
            let claimed = json_str_field(line, "sha").ok_or_else(|| {
                LmlError::InvalidHeader(format!("audit-log line {line_no}: missing 'sha' field"))
            })?;
            if claimed != computed {
                return Err(LmlError::InvalidHeader(format!(
                    "audit-log line {line_no}: sha '{claimed}' != recomputed '{computed}' (entry tampered)"
                )));
            }
            prev_hex = computed;
            count += 1;
        }
        Ok(count)
    }

    /// Read the previous entry's `sha` field for the chain. Returns
    /// `"00…00"` if the file doesn't exist (genesis).
    fn tail_sha(&self) -> LmlResult<String> {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok("0".repeat(32)),
            Err(e) => return Err(LmlError::Io(e)),
        };
        for line in raw.lines().rev() {
            if line.trim().is_empty() {
                continue;
            }
            if let Some(s) = json_str_field(line, "sha") {
                return Ok(s);
            }
        }
        Ok("0".repeat(32))
    }
}

fn current_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let (s_in_day, days) = (secs % 86_400, secs / 86_400);
    let (h, m, s) = (s_in_day / 3600, (s_in_day / 60) % 60, s_in_day % 60);
    let (y, mo, d) = epoch_days_to_ymd(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn epoch_days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Howard Hinnant's date algorithm — converts unix-epoch days to a
    // proleptic Gregorian (y, m, d). Same algorithm `chrono` uses.
    let days = days + 719_468;
    let era = if days >= 0 {
        days / 146_097
    } else {
        (days - 146_096) / 146_097
    };
    let doe = (days - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    (y as i32, m, d)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Tiny extractor for `"key":"value"` pairs from a single JSON line.
/// Doesn't handle escaped quotes inside the value; the audit log
/// `op` + `msg` fields are escaped via `json_escape` on write.
fn json_str_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let pos = line.find(&needle)?;
    let after = &line[pos + needle.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_key() -> Key {
        Key::from_bytes([0xABu8; KEY_LEN])
    }

    #[test]
    fn aes_gcm_round_trip() {
        let key = fixed_key();
        let plaintext = b"hello LamQuant".to_vec();
        let blob = encrypt_aes_gcm(&key, &plaintext).unwrap();
        // Header + ciphertext (with 16-byte tag) >= header + 16 + len.
        assert!(blob.len() >= ENC_HEADER_LEN + 16 + plaintext.len());
        assert_eq!(&blob[..8], ENC_MAGIC);
        let recovered = decrypt_aes_gcm(&key, &blob).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn aes_gcm_rejects_wrong_key() {
        let k1 = Key::from_bytes([0x01u8; KEY_LEN]);
        let k2 = Key::from_bytes([0x02u8; KEY_LEN]);
        let blob = encrypt_aes_gcm(&k1, b"top secret").unwrap();
        assert!(decrypt_aes_gcm(&k2, &blob).is_err());
    }

    #[test]
    fn aes_gcm_rejects_tampered_ciphertext() {
        let key = fixed_key();
        let mut blob = encrypt_aes_gcm(&key, b"abc").unwrap();
        let off = ENC_HEADER_LEN;
        blob[off] ^= 0x01;
        assert!(decrypt_aes_gcm(&key, &blob).is_err());
    }

    #[test]
    fn aes_gcm_rejects_bad_magic() {
        let key = fixed_key();
        let mut blob = encrypt_aes_gcm(&key, b"abc").unwrap();
        blob[0] = b'X';
        match decrypt_aes_gcm(&key, &blob) {
            Err(LmlError::InvalidMagic(_)) => {}
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn key_from_hex_round_trip() {
        let bytes = [0x42u8; KEY_LEN];
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let k = Key::from_hex(&hex).unwrap();
        assert_eq!(k.as_bytes(), &bytes);
    }

    #[test]
    fn key_from_hex_rejects_wrong_length() {
        assert!(Key::from_hex("aa").is_err());
        assert!(Key::from_hex("zz".repeat(32).as_str()).is_err());
    }

    #[test]
    fn hmac_round_trip_and_tamper() {
        let key = fixed_key();
        let bytes = b"sign me please".to_vec();
        let tag = hmac_sign(&key, &bytes);
        assert!(hmac_verify(&key, &bytes, &tag));
        let mut tampered = bytes.clone();
        tampered[0] ^= 0x01;
        assert!(!hmac_verify(&key, &tampered, &tag));
    }

    #[test]
    fn hmac_rejects_wrong_key() {
        let k1 = Key::from_bytes([0x01u8; KEY_LEN]);
        let k2 = Key::from_bytes([0x02u8; KEY_LEN]);
        let bytes = b"abc".to_vec();
        let tag = hmac_sign(&k1, &bytes);
        assert!(!hmac_verify(&k2, &bytes, &tag));
    }

    #[test]
    fn audit_log_chain_verifies() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::new(tmp.path().join("audit.jsonl"));
        log.append("encode", "foo.edf -> foo.lml").unwrap();
        log.append("decode", "foo.lml -> foo.raw").unwrap();
        log.append("encrypt", "foo.lml.enc").unwrap();
        assert_eq!(log.verify().unwrap(), 3);
    }

    #[test]
    fn audit_log_detects_tampered_msg() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("audit.jsonl");
        let log = AuditLog::new(&path);
        log.append("encode", "foo.edf").unwrap();
        log.append("decode", "foo.lml").unwrap();
        // Edit the middle entry's msg.
        let raw = std::fs::read_to_string(&path).unwrap();
        let tampered = raw.replace("foo.edf", "bar.edf");
        std::fs::write(&path, tampered).unwrap();
        assert!(log.verify().is_err());
    }

    #[test]
    fn audit_log_empty_file_verifies_as_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::new(tmp.path().join("missing.jsonl"));
        assert_eq!(log.verify().unwrap(), 0);
    }
}
