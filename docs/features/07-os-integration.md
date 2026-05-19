# OS Integration

> Make `.lma` browse like a normal directory from the desktop file
> manager. Three coexisting paths: FUSE mount via `lmafs`, MIME +
> Open-With handler, KDE Ark Kerfuffle plugin. For the underlying
> wire format used by these integrations (`lml ls --long`), see
> [Browse / Inspect](./05-browse-inspect.md).

This bucket links liberally to
[`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md)
rather than duplicating the full install walkthrough.

## At a glance

| Feature | Path / handler | Status | First shipped | Notes |
|---|---|---|---|---|
| FUSE filesystem `lmafs` | `lmafs <archive> <mountpoint>` | shipped | v1.1 (U) | Dolphin/Nautilus/Thunar/Caja/PCManFM/Finder browse |
| FUSE foreground mode | `--foreground` / `-f` | shipped | v1.1 | systemd / Docker / debugging |
| FUSE allow-other | `--allow-other` | shipped | v1.1 | Cross-user access |
| FUSE strict-decode default | (default) | shipped | v1.4 | Codec failure → `EIO` to kernel |
| FUSE raw-fallback escape hatch | `--allow-raw-fallback` | shipped | v1.4 | Forensic triage of pre-v1.1 archives |
| FUSE directory tree | (automatic) | shipped | v1.4 | Nested paths become real subdirectories |
| `application/x-lma` MIME type | `installer/install-mime.sh` | shipped | v1.1 | Magic-byte glob + extension |
| `.desktop` Open-With handler | `installer/lamquant-mime.desktop` | shipped | v1.1 | Right-click → "Open with LamQuant" |
| `lma-open` dispatcher | `installer/lma-open` | shipped | v1.4 | flock + systemd-run scope spawn |
| KDE Ark Kerfuffle plugin | `installer/ark-plugin/` | shipped | v1.4 | C++/Qt6, read-only, shells out to `lml` |
| Plugin install script | `installer/install-ark-plugin.sh` | shipped | v1.4 | distro-agnostic preflight + ABI drift sentinel |
| macOS macFUSE support | (docs only) | shipped | v1.3 | `brew install --cask macfuse` |
| Windows WinFsp support | (docs only) | shipped | v1.3 | WinFsp open-source FUSE-compat |
| libarchive plugin | (NOT FEASIBLE) | not shipped | — | libarchive has no runtime plugin API for new formats |

## Commands

### `lmafs`

FUSE filesystem that mounts a LamQuant `.lma` archive as a read-only
directory. Once mounted, every file manager that walks normal
directories sees the archive as a folder: browse, copy, drag-and-drop,
double-click-to-open all work without any plugin or shell extension.

The whole implementation is ~400 lines of pure Rust in
`crates/lmafs/src/main.rs`. No closed-source helpers, no third-party
drivers. Audit by reading the source.

Synopsis:
```
lmafs [OPTIONS] <ARCHIVE> <MOUNTPOINT>
```

Examples:
```
# Mount + browse in Dolphin/Nautilus/etc.
mkdir -p /tmp/foo-mount
lmafs recording.lma /tmp/foo-mount

# ... then open /tmp/foo-mount in any file manager
fusermount -u /tmp/foo-mount   # unmount when done

# Foreground mode (systemd / Docker / debugging)
lmafs --foreground recording.lma /tmp/foo-mount

# Forensic triage of pre-v1.1 archive (raw bytes on decode failure)
lmafs --allow-raw-fallback old-archive.lma /tmp/triage-mount
```

Directory tree: LMA manifest entries can carry slash-separated paths
(`chb06/chb06_01.edf`, `sleep-telemetry/ST7011J0-PSG.edf`). FUSE
filenames can't contain `/`, so lmafs builds a synthetic directory
tree at mount time. Each unique path prefix becomes a synthetic
directory inode; file entries are leaves.

Read-only by intent — no `--rw` flag. LMA is append-only on disk;
mutations go through `lml append` with WAL guarantees ([Archive Ops](./04-archive-ops.md)).

#### lmafs flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `<ARCHIVE>` | positional | — | Path to the `.lma` to mount |
| `<MOUNTPOINT>` | positional | — | Mount point directory (must exist, owned by invoking user) |
| `-f`, `--foreground` | bool | false | Don't daemonize; logs to stderr |
| `--allow-other` | bool | false | Allow other users to access the mount |
| `--allow-raw-fallback` | bool | false | Return raw stored LML bytes on codec failure (default: `EIO`) |

#### Strict-decode default (v1.4)

By default, a codec failure (corrupt LML entry, version skew,
malformed metadata) surfaces as `EIO` to the kernel rather than
silently returning raw stored bytes. A broken `.lml` entry shows as
unreadable in the file manager — the correct behavior for clinical
data where silent corruption is worse than a visible error.

`--allow-raw-fallback` opts into the legacy behavior. Pre-v1.1
archives with known CRC-mismatched `Method::Lml` entries return raw
LML wire bytes on fallback; applications that expect EDF input will
silently mis-interpret. Use only for forensic triage of legacy
archives.

## Installation paths

| Component | `--user` install | `--system` install |
|---|---|---|
| `lmafs` binary | `~/.local/bin/lmafs` | `/usr/local/bin/lmafs` |
| MIME XML | `~/.local/share/mime/packages/application-x-lma.xml` | `/usr/share/mime/packages/application-x-lma.xml` |
| `.desktop` entry | `~/.local/share/applications/lamquant-mime.desktop` | `/usr/share/applications/lamquant-mime.desktop` |
| Ark Kerfuffle plugin | `~/.local/lib/qt6/plugins/kerfuffle/` | `/usr/lib/qt6/plugins/kerfuffle/` |
| `lma-open` dispatcher | `~/.local/bin/lma-open` | `/usr/local/bin/lma-open` |

Per-user paths require no sudo. System paths require root and
register globally for every user on the machine.

Full walkthroughs in
[`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md).

## MIME registration (`install-mime.sh`)

Registers `application/x-lma` as a known MIME type. File managers
recognise `.lma` files by content (magic-byte match on the `LMA1`
header) AND by extension, show the package-x-generic icon, and surface
LamQuant in the right-click "Open with" menu.

What it installs:
- `application-x-lma.xml` (shared-mime-info: magic-byte glob + `*.lma`
  extension)
- `lamquant-mime.desktop` (Open-With handler that runs the
  `lma-open` dispatcher)

Run `installer/install-mime.sh --user` or `--system`. Re-runs are
no-clobber (v1.4 hardening) — your `xdg-mime default` survives.

## `lma-open` dispatcher (v1.4)

The Open-With handler in `lamquant-mime.desktop` invokes
`installer/lma-open <file>` rather than spawning `lmafs` directly.
This wrapper:

1. **Magic-byte check** — confirms `LMA1` magic before mounting
   (cheap; catches misnamed files)
2. **Flock guard** — exclusive flock prevents KDE / GNOME from
   double-firing the launcher on a fast double-click
3. **systemd-run scope spawn** — `systemd-run --user --scope lmafs ...`
   detaches the mount from the desktop launcher scope, so the FUSE
   process survives `xdg-open`'s teardown
4. **Stale-mount recovery** — walks `/proc/self/mountinfo` to detect
   abandoned mountpoints from previous crashes; lazy-unmounts if found
5. **Synchronous `xdg-open`** — opens the mountpoint in the file
   manager and waits, so the file manager process survives the
   launcher scope teardown

The wrapper is shell-only — bash parameter expansion does the
`Exec=` rewrite (no `sed`), and the script is ~150 lines you can
audit before installing.

## KDE Ark Kerfuffle plugin (v1.4)

For KDE users who want `.lma` archives to open natively in Ark (no
FUSE mount, no terminal). C++/Qt6 plugin subclasses
`Kerfuffle::CliInterface` and shells out to `lml`:

- **List** — `lml ls --long <archive>` (`#lml-ls schema=1` TSV)
- **Extract** — `lml extract <archive> -o <dir>`
- **Read-only** — no write / append / delete

Install via `installer/install-ark-plugin.sh --user` or `--system`.
The script:

- distro-agnostic preflight (`pacman` / `dpkg` / `rpm` / `pkg` detected)
- `--system` prefix corrected to `/usr` (not `/usr/local`)
- compatibility check that `lml ls --long` is supported
- Ark ABI drift sentinel: warns if the installed Ark links a Kerfuffle
  ABI different from the one the plugin was built against
- Plasma env snippet printed for `--user` installs (so the per-user
  plugin path is on the `QT_PLUGIN_PATH`)

## Coexistence: FUSE/Dolphin vs Ark

Both paths can coexist; `install-mime.sh` is no-clobber on re-run.
Switch between them with `xdg-mime`:

```sh
# Dolphin path: double-click → lma-open → lmafs FUSE mount → Dolphin opens dir
xdg-mime default lamquant-mime.desktop application/x-lma

# Ark path: double-click → Ark opens .lma natively via the Kerfuffle plugin
xdg-mime default org.kde.ark.desktop application/x-lma
```

## Build dependencies

For source builds of `lmafs` and the Ark plugin:

| Distro | FUSE for lmafs | Ark plugin (Qt6 + Kerfuffle) |
|---|---|---|
| Arch / CachyOS | `fuse3` | `qt6-base ark-devel cmake extra-cmake-modules` |
| Debian / Ubuntu | `libfuse3-dev` | `qt6-base-dev libkf6kio-dev ark-devel cmake extra-cmake-modules` |
| Fedora / RHEL | `fuse3-devel` | `qt6-qtbase-devel ark-devel cmake extra-cmake-modules` |
| FreeBSD | `fusefs-libs3` | (Ark not packaged) |

The `installer/install-ark-plugin.sh` script auto-detects the
package manager and reports any missing dependencies before building.

## macOS macFUSE

`lmafs` builds against macFUSE via the same `fuser` crate. macFUSE is
a third-party kext + System Extension; install once via
[macfuse.io](https://macfuse.io):

```sh
brew install --cask macfuse        # one-time, requires reboot
cargo build --release -p lmafs
./target/release/lmafs foo.lma /Volumes/foo
# Open /Volumes/foo in Finder
umount /Volumes/foo
```

Finder Quick Look (`.qlgenerator`) integration is out-of-scope —
needs an Apple Developer ID and notarization budget. The FUSE mount
covers the browse use case.

## Windows WinFsp

`lmafs.exe` works against [WinFsp](https://winfsp.dev/) (free,
open-source, FUSE-compatible). Install WinFsp first, then:

```powershell
cargo build --release -p lmafs
.\target\release\lmafs.exe foo.lma X:
# Open X: in Explorer
fsutil dismount X:
```

Windows Explorer shell extension integration is out-of-scope — needs
MSVC + Authenticode + COM. FUSE mount covers the browse use case.

## libarchive plugin (NOT FEASIBLE)

Investigated and reclassified. libarchive has no runtime plugin API
for new container formats — only for compression filters (gzip, bzip2,
zstd, etc.). Adding LMA as a first-class container would require
forking libarchive and shipping a patched build. Not in scope.

FUSE `lmafs` and the KDE Ark Kerfuffle plugin together cover the
file-manager integration use cases on Linux. See
[`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md)
§D for the architectural decision record.

## Error cases

| Trigger | Behavior |
|---|---|
| `lmafs` mountpoint doesn't exist | "mountpoint does not exist" |
| `lmafs` mountpoint not empty | FUSE kernel refuses (standard semantics) |
| `lmafs` `.lma` file unreadable | "lmafs: failed to read manifest from <path>" |
| Codec failure inside lmafs (default strict) | `EIO` to kernel; file shows as unreadable |
| Codec failure inside lmafs (`--allow-raw-fallback`) | raw stored LML wire bytes returned |
| `lma-open` stale mount | walks `/proc/self/mountinfo`, lazy-unmounts, retries |
| `lma-open` double-fire (KDE) | second invocation hits the flock and exits silently |
| Ark Kerfuffle ABI drift | install script warns; plugin may crash Ark on load until rebuilt |

## Related

- **Other buckets**:
  - [Browse / Inspect](./05-browse-inspect.md) — `lml ls --long` is the Ark plugin wire format
  - [Decompression](./02-decompression.md) — `--allow-raw-fallback` policy
  - [Archive Ops](./04-archive-ops.md) — multi-volume archives auto-resolve through lmafs
  - [Operational](./09-operational.md) — `lmafs.service` systemd auto-mount
- **Source files**:
  - `crates/lmafs/src/main.rs` — full FUSE implementation (~760 LOC)
  - `installer/lma-open` — dispatcher script (magic-byte + flock + systemd-run + xdg-open)
  - `installer/lamquant-mime.desktop` — Open-With handler
  - `installer/install-mime.sh` — MIME registration script
  - `installer/install-ark-plugin.sh` — Ark plugin install script
  - `installer/ark-plugin/clilma.cpp` — Kerfuffle `CliInterface` subclass
  - `installer/mime/application-x-lma.xml` — shared-mime-info XML
- **Tests**:
  - `tests/integration/test_lmafs_fuse.py`
  - `tests/integration/test_lma_open_dispatcher.sh`
  - `tests/integration/test_ark_plugin_parse.py`
- **Commits**:
  - `a30474c` — FUSE `lmafs` (v1.1 U)
  - `93c41bf` — MIME registration (v1.1 9.6)
  - `9da9d1b` — Ark Kerfuffle plugin (v1.4 12.1)
  - `f584432` — lmafs strict-decode default + directory tree (v1.4)
  - `9571655` — `lma-open` hardening (v1.4 12.6)
  - `1ee9d01` — flock-guard against KDE double-firing
- **Cross-cutting docs**:
  - [`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md) — full walkthrough
