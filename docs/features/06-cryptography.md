# Cryptography

> AES-GCM encrypt / decrypt, HMAC sign / verify, OS keyring, and the
> tamper-evident audit log. For the broader integrity story (CRC,
> SHA), see [Verification](./03-verification.md).

LamQuant ships an opt-in cryptographic layer that sits on top of the
codec. The primitives are conventional: AES-256-GCM AEAD for
encryption, HMAC-SHA-256 detached signatures for authenticity, Argon2id
for password-derived keys. Encryption is **not** mandatory — the
default lossless workflow is unencrypted plaintext archives.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| AES-256-GCM encrypt | `lml encrypt` | shipped | v1.0 (7.1) | `LAMQUANT_KEY` env hex; self-describing blob |
| AES-256-GCM decrypt | `lml decrypt` | shipped | v1.0 (7.1) | Errors on bad magic / auth-tag / wrong key |
| Argon2id password KDF | `--password` | shipped | v1.2 (P) | OWASP defaults (m=64 MiB, t=3, p=1) |
| HMAC-SHA-256 sign | `lml sign` | shipped | v1.0 (7.2) | Detached 32-byte tag sidecar |
| HMAC-SHA-256 verify | `lml verify-signature` | shipped | v1.0 (7.2) | Reads `<input>.hmac` |
| OS keyring | `--features keyring` | shipped | v1.0 (7.3) | Linux/macOS/Windows via `keyring` crate |
| Secure key drop | (automatic) | shipped | v1.0 (7.4) | `zeroize` on 32-byte key buffer |
| Tamper-evident audit log | `lml audit-log` | shipped | v1.0 (7.6) | SHA-chained JSONL |
| Release notarize / Authenticode | (CI lane) | shipped | v1.0 (7.7) | Opt-in via repo secrets |
| Encrypt-manifest (header) | (planned) | not shipped | — | LMA wire-format v2.0 follow-up; out of scope today |

## Commands

### `lml encrypt`

AES-256-GCM encrypt a file. Output blob is self-describing:

```
[8 bytes]    Magic: b"LMLCRYPT"
[1 byte]     Version
[12 bytes]   Nonce
[variable]   GCM ciphertext
[16 bytes]   Auth tag (appended by AES-GCM)
```

Key sourcing (mutually exclusive):

1. **`LAMQUANT_KEY` env** — 64-char hex = 32 raw bytes. Scripted use.
2. **`--password`** — Argon2id derives the 32-byte key from a password.
   Salt + Argon2 params written to `<output>.lmcrypt.header` sidecar.
   Reads password from `LAMQUANT_PASSWORD` env or prompts interactively
   without echo.

Synopsis:
```
lml encrypt <INPUT> -o <OUTPUT> [--password] [--force]
```

Examples:
```
# Hex key from env (CI / scripts)
export LAMQUANT_KEY=$(openssl rand -hex 32)
lml encrypt recording.lma -o recording.lma.enc

# Argon2id password KDF (interactive prompt)
lml encrypt recording.lma -o recording.lma.enc --password

# Same, password from env (CI with secret-store integration)
LAMQUANT_PASSWORD='hunter2' lml encrypt recording.lma -o recording.lma.enc --password
```

### `lml decrypt`

AES-256-GCM decrypt + authenticate. Errors on:

- Bad `LMLCRYPT` magic / version
- Auth-tag mismatch (wrong key OR tampered ciphertext)
- Truncated input

Key sourcing mirrors `encrypt`: `LAMQUANT_KEY` env or `--password`.
For `--password`, reads the salt + Argon2 params from
`<input>.lmcrypt.header` and re-derives the same key.

Synopsis:
```
lml decrypt <INPUT> -o <OUTPUT> [--password] [--force]
```

Example:
```
lml decrypt recording.lma.enc -o recording.lma --password
```

### `lml sign`

HMAC-SHA-256 sign a file. Detached 32-byte tag written to
`<input>.hmac` (or the path passed to `-o`). Key from `LAMQUANT_KEY`.

Synopsis:
```
lml sign <INPUT> [-o <TAG_PATH>] [--force]
```

Example:
```
lml sign recording.lma          # writes recording.lma.hmac
```

### `lml verify-signature`

Verify the HMAC-SHA-256 tag of a file. Reads the 32-byte tag from
`<input>.hmac` (or the path passed to `--tag`). Constant-time tag
comparison.

Synopsis:
```
lml verify-signature <INPUT> [--tag <PATH>]
```

Example:
```
lml verify-signature recording.lma
```

### `lml audit-log`

See [Verification](./03-verification.md) for the audit log surface.
Briefly:

```
lml audit-log append --log <PATH> --op <OP> --msg <MSG>
lml audit-log verify --log <PATH>
```

SHA-chained JSONL where each entry's SHA depends on the previous.
Tampering with history breaks the chain. Useful for chain-of-custody
in clinical / regulated environments.

## `.lmcrypt.header` sidecar format

The Argon2id KDF needs the salt + cost parameters to re-derive the
same key. Storing them alongside the ciphertext means the operator
only needs to remember the password.

```
[4 bytes]    Magic: b"LMHP"
[1 byte]     Version (1)
[16 bytes]   Salt
[4 bytes]    Argon2 memory (KiB) — default 65536 (64 MiB)
[4 bytes]    Argon2 time cost — default 3
[1 byte]     Argon2 parallelism — default 1
```

OWASP-recommended defaults at v1.2 ship time. Future versions may
raise the parameters; the version byte is the migration knob.

Sidecar lives at `<ciphertext_path>.lmcrypt.header`. Deleting it makes
the ciphertext undecryptable even with the password.

## OS keyring integration (`--features keyring`)

Optional Cargo feature that wires the `keyring` crate:

- Linux: Secret Service (kwallet / gnome-keyring)
- macOS: Keychain
- Windows: Credential Manager

API:

```rust
let key = Key::from_keyring("lamquant", "default")?;   // load
key.store_in_keyring("lamquant", "default")?;          // save
```

`Key::Drop` impl zeroes the 32-byte buffer via `zeroize`. No
plaintext key sits in process memory after the value goes out of
scope.

The CLI does not currently expose keyring loading directly — the
feature is library-only as of v1.2. Custom integrations (TUI, GUI, Tauri
app) call the API.

## Secure key drop (`zeroize`)

The `Key` newtype's `Drop` impl runs `zeroize::Zeroize::zeroize()` on
the 32-byte buffer. The compiler is prevented from optimising the
zero-write away (the whole point of the `zeroize` crate).

This closes the "key recoverable from process memory" footgun. Combined
with `--features keyring`, the chain is: keyring backend → in-process
`Key` (zeroized on drop) → never written to disk.

## HMAC sign vs AES-GCM encrypt

| Property | `lml sign` | `lml encrypt` |
|---|---|---|
| Confidentiality | No — file stays plaintext | Yes — AES-GCM cipherstream |
| Authenticity | Yes — HMAC tag proves issuer | Yes — GCM auth tag covers ciphertext |
| Tampering detection | Yes (HMAC mismatch) | Yes (auth-tag mismatch) |
| Key material | 32 bytes from `LAMQUANT_KEY` | 32 bytes from `LAMQUANT_KEY` or Argon2id(password) |
| Output | `<input>.hmac` sidecar | `<input>.enc` (LMLCRYPT blob) + optional `.lmcrypt.header` |

Sign + verify is the right primitive for "did this archive come from
us?" — verifiers only need the public key (well, the shared HMAC key
in practice). Encrypt is the right primitive for "nobody but the
keyholder can read this".

## Release-pipeline crypto (Phase 7.7)

CI release lane has opt-in slots for:

- **macOS notarization** — Apple Developer ID + `notarytool` upload.
  Triggered by `APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID` repo
  secrets. Skipped on PRs.
- **Windows Authenticode** — `signtool.exe` with a code-signing cert.
  Triggered by `WINDOWS_CERT_PFX_B64` / `WINDOWS_CERT_PASSWORD` repo
  secrets.

Both are wired into `.github/workflows/release.yml`. Unsigned builds
are the default; signing is opt-in for projects with the budget.

## Flags

| Flag | Type | Default | Subcommand | Description |
|---|---|---|---|---|
| `-o`, `--output <PATH>` | path | — | most | Output file path |
| `--force` | bool | false | most | Overwrite existing output |
| `--password` | bool | false | `encrypt` / `decrypt` | Use Argon2id KDF + `LAMQUANT_PASSWORD` env / prompt |
| `--tag <PATH>` | path | `<input>.hmac` | `verify-signature` | Override tag-file path |

## Environment variables

| Var | Used by | Notes |
|---|---|---|
| `LAMQUANT_KEY` | `encrypt` / `decrypt` / `sign` / `verify-signature` | 64-char hex = 32 raw bytes. Mutually exclusive with `--password` for `encrypt` / `decrypt`. |
| `LAMQUANT_PASSWORD` | `encrypt` / `decrypt` w/ `--password` | If unset, prompt interactively without echo |

## Error cases

| Trigger | Error |
|---|---|
| `decrypt` wrong key | "AES-GCM auth-tag mismatch" |
| `decrypt` corrupted ciphertext | "AES-GCM auth-tag mismatch" (indistinguishable from wrong key — by design) |
| `decrypt` bad magic | "not a LMLCRYPT blob" |
| `--password` without sidecar at decrypt | "missing `.lmcrypt.header` sidecar" |
| `LAMQUANT_KEY` not 64 hex chars | "LAMQUANT_KEY must be 64 hex chars (32 bytes)" |
| `--password` together with `LAMQUANT_KEY` env set | refuse — pick one |
| `verify-signature` HMAC mismatch | exit code 1, "HMAC tag mismatch" |
| `audit-log verify` chain break | reports the first broken-chain index |

## Related

- **Other buckets**:
  - [Verification](./03-verification.md) — CRC / SHA layered checks, audit log details
  - [Compression](./01-compression.md) — encrypt happens AFTER encode (operate on `.lma`)
  - [Build / Release](./10-build-release.md) — notarize + Authenticode CI lanes
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:733` — `Encrypt`
  - `lamquant-core/src/bin/lml.rs:754` — `Decrypt`
  - `lamquant-core/src/bin/lml.rs:772` — `Sign`
  - `lamquant-core/src/bin/lml.rs:784` — `VerifySignature`
  - `lamquant-core/src/bin/lml.rs:792` — `AuditLog`
  - `lamquant-core/src/security.rs` — Key, AES-GCM, HMAC, AuditLog, keyring impls
- **Tests**:
  - `tests/integration/test_encrypt_roundtrip.py`
  - `tests/integration/test_argon2_password.py`
  - `tests/integration/test_sign_verify.py`
  - `tests/integration/test_audit_log.py`
- **Commits**:
  - `b0f2ff8` — `--password` Argon2id KDF (v1.2 P)
  - Phase 7.1 / 7.2 / 7.4 — AES-GCM + HMAC + zeroize
  - Phase 7.6 — tamper-evident audit log
  - Phase 7.7 — notarize + Authenticode CI slots
- **Cross-cutting docs**:
  - [`../COMPLIANCE.md`](../COMPLIANCE.md) — regulatory framing for the crypto stack
  - [`../FAQ.md`](../FAQ.md) — common encrypt / key-handling questions
