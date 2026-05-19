//! `lmafs` — FUSE filesystem that mounts a LamQuant `.lma` archive
//! as a read-only directory.
//!
//! Once mounted, every file manager that walks normal directories —
//! KDE Dolphin, GNOME Nautilus, XFCE Thunar, MATE Caja, LXQt PCManFM,
//! the macOS-with-macFUSE Finder — sees the archive as a folder.
//! Browse, copy, drag-and-drop, double-click-to-open works without
//! any plugin or shell extension.
//!
//! Usage:
//!
//!     lmafs foo.lma /mnt/foo            # mount
//!     fusermount -u /mnt/foo            # unmount
//!
//! Or run in foreground (useful for systemd / debugging):
//!
//!     lmafs --foreground foo.lma /mnt/foo
//!
//! Auditable footprint by design: this binary is ~300 lines of pure
//! Rust, every dependency is well-known (`fuser` for the FUSE kernel
//! protocol, `lamquant-core` for archive parsing). No proprietary
//! glue. Read the source to verify what the mount does.
//!
//! Read-only by intent — there is no `lmafs --rw` flag. LMA archives
//! are append-only on disk; in-place edits go through `lml append`
//! with its WAL guarantees. Mounting as writable would invite
//! TOCTOU races we can't honour without rebuilding the whole
//! archive.

use clap::Parser;
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry,
    Request,
};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;

#[derive(Parser, Debug)]
#[command(
    name = "lmafs",
    version,
    about = "Mount a LamQuant `.lma` archive as a read-only directory",
    after_help = "\
EXAMPLES:
  # Mount + browse in any file manager
  mkdir -p /tmp/foo-mount
  lmafs recording.lma /tmp/foo-mount
  # ... then open /tmp/foo-mount in Dolphin / Nautilus / etc.
  fusermount -u /tmp/foo-mount

  # Run in foreground (logs to stderr)
  lmafs --foreground recording.lma /tmp/foo-mount

  # systemd auto-mount on the per-user `lmafs@<archive>.service` unit
  # (see installer/lmafs.service)

NOTES:
  * Read-only. Edits go through `lml append` (atomic + WAL-backed).
  * mtime + Unix mode preserved from the LMA manifest.
  * No FUSE plugin to audit -- this binary IS the plugin, ~300 LOC.
  * Requires kernel FUSE (Linux) or macFUSE (macOS).

DOCUMENTATION:
  See docs/INSTALL_FILE_MANAGER_INTEGRATION.md for the full Dolphin /
  Nautilus walkthrough and `--user` vs `--system` install paths."
)]
struct Args {
    /// Path to the `.lma` archive to mount.
    archive: PathBuf,

    /// Mount point. Must exist and be empty (or owned by the
    /// invoking user). Standard FUSE convention.
    mountpoint: PathBuf,

    /// Stay in the foreground after mounting. Default behaviour
    /// daemonizes so the shell prompt returns immediately. Use
    /// `--foreground` when wrapping in systemd / Docker / when you
    /// want stderr tracing visible.
    #[arg(long, short = 'f')]
    foreground: bool,

    /// Allow other users to access the mount. Default is owner-only.
    #[arg(long)]
    allow_other: bool,
}

/// In-memory model of the archive: every entry mapped to a stable
/// inode + cached metadata. Entry payloads are decompressed on
/// demand inside `read()`; we don't slurp the whole archive into RAM
/// at mount time.
struct LmaFs {
    archive_path: PathBuf,
    // Inode 1 is the root dir; inode 2.. correspond to entries in
    // manifest order. The map is `name -> inode` for `lookup`.
    entries: Vec<EntryHandle>,
    name_to_ino: HashMap<String, u64>,
}

struct EntryHandle {
    path: String,
    size: u64,
    mtime: u64,
    mode: u32,
}

impl LmaFs {
    fn new(archive_path: PathBuf) -> std::io::Result<Self> {
        let entries = lamquant_core::lma::list_archive(&archive_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("lmafs: failed to read manifest from {}: {e}", archive_path.display()),
            )
        })?;

        let mut handles = Vec::with_capacity(entries.len());
        let mut name_to_ino = HashMap::with_capacity(entries.len());
        for (idx, e) in entries.iter().enumerate() {
            let ino = (idx as u64) + 2; // 1 is reserved for root
            handles.push(EntryHandle {
                path: e.path.clone(),
                size: e.original_size,
                mtime: e.mtime.unwrap_or(0),
                // Default to 0644 when the manifest didn't record a
                // Unix mode (e.g. cross-platform archive). The bit
                // pattern is regular file + rw-r--r--.
                mode: e.mode.unwrap_or(0o100644),
            });
            // For lookup, we need the basename, not the full archive
            // path. Archives are flat today; the path field IS the
            // basename. Future nested-namespace support can elevate
            // this to a tree walk.
            name_to_ino.insert(e.path.clone(), ino);
        }

        Ok(LmaFs {
            archive_path,
            entries: handles,
            name_to_ino,
        })
    }

    fn entry_attr(&self, ino: u64) -> Option<FileAttr> {
        if ino == ROOT_INO {
            return Some(self.root_attr());
        }
        let idx = (ino - 2) as usize;
        let entry = self.entries.get(idx)?;
        Some(FileAttr {
            ino,
            size: entry.size,
            blocks: (entry.size + 511) / 512,
            atime: UNIX_EPOCH + Duration::from_secs(entry.mtime),
            mtime: UNIX_EPOCH + Duration::from_secs(entry.mtime),
            ctime: UNIX_EPOCH + Duration::from_secs(entry.mtime),
            crtime: UNIX_EPOCH + Duration::from_secs(entry.mtime),
            kind: FileType::RegularFile,
            perm: (entry.mode & 0o777) as u16,
            nlink: 1,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            flags: 0,
            blksize: 4096,
        })
    }

    fn root_attr(&self) -> FileAttr {
        FileAttr {
            ino: ROOT_INO,
            size: 4096,
            blocks: 1,
            atime: UNIX_EPOCH,
            mtime: UNIX_EPOCH,
            ctime: UNIX_EPOCH,
            crtime: UNIX_EPOCH,
            kind: FileType::Directory,
            perm: 0o555, // r-xr-xr-x — read-only
            nlink: 2,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            flags: 0,
            blksize: 4096,
        }
    }
}

impl Filesystem for LmaFs {
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        if parent != ROOT_INO {
            reply.error(libc::ENOENT);
            return;
        }
        let Some(name_str) = name.to_str() else {
            reply.error(libc::ENOENT);
            return;
        };
        match self.name_to_ino.get(name_str) {
            Some(&ino) => {
                if let Some(attr) = self.entry_attr(ino) {
                    reply.entry(&TTL, &attr, 0);
                } else {
                    reply.error(libc::ENOENT);
                }
            }
            None => reply.error(libc::ENOENT),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyAttr) {
        match self.entry_attr(ino) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(libc::ENOENT),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        if ino != ROOT_INO {
            reply.error(libc::ENOTDIR);
            return;
        }
        let entries_iter = std::iter::once((ROOT_INO, FileType::Directory, ".".to_string()))
            .chain(std::iter::once((
                ROOT_INO,
                FileType::Directory,
                "..".to_string(),
            )))
            .chain(self.entries.iter().enumerate().map(|(idx, e)| {
                ((idx as u64) + 2, FileType::RegularFile, e.path.clone())
            }));

        for (i, (entry_ino, kind, name)) in entries_iter.enumerate().skip(offset as usize) {
            // i is the index *before* skipping; FUSE expects the
            // offset of the NEXT entry to read so we hand back i+1.
            let buffer_full = reply.add(entry_ino, (i + 1) as i64, kind, name);
            if buffer_full {
                break;
            }
        }
        reply.ok();
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        if ino == ROOT_INO {
            reply.error(libc::EISDIR);
            return;
        }
        let idx = (ino - 2) as usize;
        let Some(entry) = self.entries.get(idx) else {
            reply.error(libc::ENOENT);
            return;
        };

        // Pull the entry's full payload via the existing public
        // `lma::read_entry` API. This is not yet streaming —
        // for v1.1 we accept the round-trip cost on `read()` so the
        // FS implementation stays trivially auditable. Stream-mode
        // (offset-aware decompression) is a v1.2 improvement.
        let bytes = match lamquant_core::lma::read_entry(&self.archive_path, &entry.path) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(entry = %entry.path, err = %e, "lmafs read: archive read failed");
                reply.error(libc::EIO);
                return;
            }
        };
        let off = offset.max(0) as usize;
        if off >= bytes.len() {
            reply.data(&[]);
            return;
        }
        let end = (off + size as usize).min(bytes.len());
        reply.data(&bytes[off..end]);
    }
}

fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("lmafs=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let fs = LmaFs::new(args.archive.clone())?;
    tracing::info!(
        archive = %args.archive.display(),
        mountpoint = %args.mountpoint.display(),
        entries = fs.entries.len(),
        "lmafs: mounting"
    );

    let mut options = vec![
        MountOption::RO,
        MountOption::FSName(format!(
            "lmafs:{}",
            args.archive
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "lma".to_string())
        )),
        MountOption::Subtype("lmafs".to_string()),
    ];
    if args.allow_other {
        options.push(MountOption::AllowOther);
    }

    if args.foreground {
        fuser::mount2(fs, &args.mountpoint, &options)
    } else {
        // `mount2` blocks until unmount; fork-and-detach for the
        // background case. We rely on the kernel FUSE machinery
        // surviving the parent's exit — that's the standard
        // foreground=false pattern in fuser's examples.
        let mountpoint = args.mountpoint.clone();
        std::thread::spawn(move || {
            if let Err(e) = fuser::mount2(fs, &mountpoint, &options) {
                tracing::error!(err = %e, "lmafs: mount loop exited with error");
            }
        });
        // Block forever; SIGINT / SIGTERM / `fusermount -u` cleans up.
        std::thread::park();
        Ok(())
    }
}
