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
//! Auditable footprint by design: this binary is ~400 lines of pure
//! Rust, every dependency is well-known (`fuser` for the FUSE kernel
//! protocol, `lamquant-core` for archive parsing). No proprietary
//! glue. Read the source to verify what the mount does.
//!
//! Read-only by intent — there is no `lmafs --rw` flag. LMA archives
//! are append-only on disk; in-place edits go through `lml append`
//! with its WAL guarantees. Mounting as writable would invite
//! TOCTOU races we can't honour without rebuilding the whole
//! archive.
//!
//! Directory tree: LMA manifest entries can carry slash-separated
//! paths (`chb06/chb06_01.edf`, `sleep-telemetry/ST7011J0-PSG.edf`).
//! FUSE filenames cannot contain `/`, so we build a synthetic
//! directory tree at mount time. Each unique path prefix becomes a
//! synthetic directory inode; file entries are leaves. `readdir` /
//! `lookup` walk one level at a time, file managers see a proper
//! folder hierarchy.

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
  * Nested paths in the archive manifest become real subdirectories.
  * No FUSE plugin to audit -- this binary IS the plugin, ~400 LOC.
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

/// File-entry metadata pulled verbatim from the LMA manifest.
struct EntryHandle {
    /// Full archive path (e.g. `chb06/chb06_01.edf`). Used by
    /// `lma::read_entry` to fetch the payload.
    path: String,
    size: u64,
    mtime: u64,
    mode: u32,
}

/// An inode is either a synthetic directory (built from path prefixes
/// at mount time) or a file (one of the manifest entries).
enum INode {
    Dir {
        /// (child_ino, basename, kind). Sorted by basename for
        /// deterministic readdir order.
        children: Vec<(u64, String, FileType)>,
    },
    File {
        /// Index into `LmaFs::entries`.
        entry_idx: usize,
    },
}

/// In-memory model of the archive: tree of synthetic dirs +
/// leaf files. Entry payloads are decompressed on demand inside
/// `read()`; we don't slurp the whole archive into RAM at mount time.
struct LmaFs {
    archive_path: PathBuf,
    /// All file entries from the manifest, flat. Indexed by
    /// `INode::File.entry_idx`.
    entries: Vec<EntryHandle>,
    /// inode -> dir or file. ROOT_INO is always a Dir.
    inodes: HashMap<u64, INode>,
}

impl LmaFs {
    fn new(archive_path: PathBuf) -> std::io::Result<Self> {
        let manifest = lamquant_core::lma::list_archive(&archive_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("lmafs: failed to read manifest from {}: {e}", archive_path.display()),
            )
        })?;

        // Phase 1: collect file entries verbatim.
        let mut entries: Vec<EntryHandle> = Vec::with_capacity(manifest.len());
        for e in &manifest {
            entries.push(EntryHandle {
                path: e.path.clone(),
                size: e.original_size,
                mtime: e.mtime.unwrap_or(0),
                // Default to 0644 when the manifest didn't record a
                // Unix mode (e.g. cross-platform archive). The bit
                // pattern is regular file + rw-r--r--.
                mode: e.mode.unwrap_or(0o100644),
            });
        }

        // Phase 2: build the synthetic directory tree.
        //
        // For each entry path, split on `/`, walking down from the
        // root. Each unique prefix becomes a directory inode; the
        // final segment becomes a file inode pointing back to its
        // entry. Slash-free archives degenerate to a flat tree under
        // the root (same as the v1.1 behaviour).
        //
        // Inode allocation:
        //   ROOT = 1
        //   then incrementing counter for every new dir + file.
        let mut inodes: HashMap<u64, INode> = HashMap::new();
        inodes.insert(ROOT_INO, INode::Dir { children: Vec::new() });

        // `dir_lookup[parent_ino][segment_basename] = child_dir_ino`.
        // Avoids re-walking children Vec on every insert.
        let mut dir_lookup: HashMap<(u64, String), u64> = HashMap::new();
        let mut next_ino: u64 = 2;

        for (entry_idx, entry) in entries.iter().enumerate() {
            // Normalise: strip any leading `/`, ignore empty segments
            // (defensive — well-formed manifests never have them, but
            // a malformed archive shouldn't crash the mount).
            let segments: Vec<&str> = entry
                .path
                .split('/')
                .filter(|s| !s.is_empty())
                .collect();
            if segments.is_empty() {
                tracing::warn!(path = %entry.path, "lmafs: skipping entry with empty path");
                continue;
            }

            // Walk down dirs, creating missing ones.
            let mut parent_ino = ROOT_INO;
            for (i, seg) in segments.iter().enumerate() {
                let is_last = i == segments.len() - 1;
                let seg_owned = (*seg).to_string();

                if is_last {
                    // Final segment = the file leaf.
                    let file_ino = next_ino;
                    next_ino += 1;
                    inodes.insert(file_ino, INode::File { entry_idx });
                    if let Some(INode::Dir { children }) = inodes.get_mut(&parent_ino) {
                        children.push((file_ino, seg_owned, FileType::RegularFile));
                    }
                } else {
                    // Intermediate dir — reuse if already created,
                    // otherwise allocate a new dir inode.
                    let key = (parent_ino, seg_owned.clone());
                    let child_ino = if let Some(&existing) = dir_lookup.get(&key) {
                        existing
                    } else {
                        let new_dir_ino = next_ino;
                        next_ino += 1;
                        inodes.insert(
                            new_dir_ino,
                            INode::Dir { children: Vec::new() },
                        );
                        dir_lookup.insert(key, new_dir_ino);
                        if let Some(INode::Dir { children }) = inodes.get_mut(&parent_ino) {
                            children.push((new_dir_ino, seg_owned, FileType::Directory));
                        }
                        new_dir_ino
                    };
                    parent_ino = child_ino;
                }
            }
        }

        // Sort children of every dir by basename for deterministic
        // readdir order (matches the order users see when running
        // `ls` on a normal directory).
        for node in inodes.values_mut() {
            if let INode::Dir { children } = node {
                children.sort_by(|a, b| a.1.cmp(&b.1));
            }
        }

        Ok(LmaFs {
            archive_path,
            entries,
            inodes,
        })
    }

    fn entry_attr(&self, ino: u64) -> Option<FileAttr> {
        let node = self.inodes.get(&ino)?;
        match node {
            INode::Dir { .. } => Some(self.dir_attr(ino)),
            INode::File { entry_idx } => {
                let entry = self.entries.get(*entry_idx)?;
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
        }
    }

    fn dir_attr(&self, ino: u64) -> FileAttr {
        FileAttr {
            ino,
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
        let Some(parent_node) = self.inodes.get(&parent) else {
            reply.error(libc::ENOENT);
            return;
        };
        let children = match parent_node {
            INode::Dir { children } => children,
            INode::File { .. } => {
                reply.error(libc::ENOTDIR);
                return;
            }
        };
        let Some(name_str) = name.to_str() else {
            reply.error(libc::ENOENT);
            return;
        };
        // Linear scan — children list is small (usually <100 entries
        // per dir in real LMA archives). Switch to BTreeMap if a
        // pathological archive surfaces with thousands of siblings.
        let child = children.iter().find(|(_, n, _)| n == name_str);
        match child {
            Some((ino, _, _)) => {
                if let Some(attr) = self.entry_attr(*ino) {
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
        let Some(node) = self.inodes.get(&ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let children = match node {
            INode::Dir { children } => children,
            INode::File { .. } => {
                reply.error(libc::ENOTDIR);
                return;
            }
        };
        let entries_iter = std::iter::once((ino, FileType::Directory, ".".to_string()))
            .chain(std::iter::once((ino, FileType::Directory, "..".to_string())))
            .chain(
                children
                    .iter()
                    .map(|(c_ino, name, kind)| (*c_ino, *kind, name.clone())),
            );

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
        let entry_idx = match self.inodes.get(&ino) {
            Some(INode::File { entry_idx }) => *entry_idx,
            Some(INode::Dir { .. }) => {
                reply.error(libc::EISDIR);
                return;
            }
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };
        let Some(entry) = self.entries.get(entry_idx) else {
            reply.error(libc::ENOENT);
            return;
        };

        // Pull the entry's *decoded* payload via `read_entry_decoded`.
        // For Store / Zstd entries this is identical to `read_entry`;
        // for `Method::Lml` it transparently reconstructs the
        // byte-identical EDF/BDF source so the file manager sees the
        // original file, not raw LML wire bytes.
        //
        // Graceful fallback: if the LML codec rejects the entry
        // (e.g. an older archive packed by a pre-v1.1 encoder whose
        // CRC scheme the current decoder no longer recognises), drop
        // back to the raw stored bytes so the user at least sees the
        // LML wire payload rather than the kernel returning EIO.
        // We log the decode error so it remains visible in `journalctl`
        // / stderr — silent fallback would mask a real codec bug.
        //
        // This is not yet streaming — we accept the full-entry
        // round-trip cost on `read()` so the FS implementation stays
        // trivially auditable. Stream-mode (offset-aware decompression
        // + cache) is a future improvement.
        let bytes = match lamquant_core::lma::read_entry_decoded(&self.archive_path, &entry.path) {
            Ok(b) => b,
            Err(decode_err) => {
                tracing::warn!(
                    entry = %entry.path,
                    err = %decode_err,
                    "lmafs read: decoded read failed -- falling back to raw payload"
                );
                match lamquant_core::lma::read_entry(&self.archive_path, &entry.path) {
                    Ok(b) => b,
                    Err(raw_err) => {
                        tracing::error!(
                            entry = %entry.path,
                            err = %raw_err,
                            "lmafs read: archive read failed (raw fallback also failed)"
                        );
                        reply.error(libc::EIO);
                        return;
                    }
                }
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
        inodes = fs.inodes.len(),
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

#[cfg(test)]
mod tests {
    //! Unit tests of the tree-building logic. We can't easily test
    //! the FUSE loop end-to-end in a unit test (would need a real
    //! mount + kernel), but we CAN verify the inode tree is built
    //! correctly from a manifest, since that's where the historical
    //! flat-readdir bug lived.
    use super::*;

    fn mk_entry(path: &str) -> EntryHandle {
        EntryHandle { path: path.to_string(), size: 1, mtime: 0, mode: 0o100644 }
    }

    /// Build a tree from manifest paths directly, bypassing the
    /// archive read. Mirrors the body of `LmaFs::new` Phase 2 so we
    /// can unit-test the tree construction logic in isolation.
    fn build_tree(paths: &[&str]) -> LmaFs {
        let entries: Vec<EntryHandle> = paths.iter().map(|p| mk_entry(p)).collect();
        let mut inodes: HashMap<u64, INode> = HashMap::new();
        inodes.insert(ROOT_INO, INode::Dir { children: Vec::new() });
        let mut dir_lookup: HashMap<(u64, String), u64> = HashMap::new();
        let mut next_ino: u64 = 2;
        for (entry_idx, entry) in entries.iter().enumerate() {
            let segments: Vec<&str> =
                entry.path.split('/').filter(|s| !s.is_empty()).collect();
            if segments.is_empty() {
                continue;
            }
            let mut parent_ino = ROOT_INO;
            for (i, seg) in segments.iter().enumerate() {
                let is_last = i == segments.len() - 1;
                let seg_owned = (*seg).to_string();
                if is_last {
                    let file_ino = next_ino;
                    next_ino += 1;
                    inodes.insert(file_ino, INode::File { entry_idx });
                    if let Some(INode::Dir { children }) = inodes.get_mut(&parent_ino) {
                        children.push((file_ino, seg_owned, FileType::RegularFile));
                    }
                } else {
                    let key = (parent_ino, seg_owned.clone());
                    let child_ino = if let Some(&e) = dir_lookup.get(&key) {
                        e
                    } else {
                        let new_dir_ino = next_ino;
                        next_ino += 1;
                        inodes
                            .insert(new_dir_ino, INode::Dir { children: Vec::new() });
                        dir_lookup.insert(key, new_dir_ino);
                        if let Some(INode::Dir { children }) = inodes.get_mut(&parent_ino) {
                            children.push((new_dir_ino, seg_owned, FileType::Directory));
                        }
                        new_dir_ino
                    };
                    parent_ino = child_ino;
                }
            }
        }
        for node in inodes.values_mut() {
            if let INode::Dir { children } = node {
                children.sort_by(|a, b| a.1.cmp(&b.1));
            }
        }
        LmaFs { archive_path: PathBuf::from("/dev/null"), entries, inodes }
    }

    fn root_children(fs: &LmaFs) -> Vec<(String, FileType)> {
        match fs.inodes.get(&ROOT_INO).unwrap() {
            INode::Dir { children } => children
                .iter()
                .map(|(_, n, k)| (n.clone(), *k))
                .collect(),
            _ => panic!("root is not a dir"),
        }
    }

    #[test]
    fn flat_archive_lists_files_at_root() {
        let fs = build_tree(&["README.txt", "rec.edf", "meta.json"]);
        let mut names: Vec<_> =
            root_children(&fs).into_iter().map(|(n, _)| n).collect();
        names.sort();
        assert_eq!(names, vec!["README.txt", "meta.json", "rec.edf"]);
    }

    #[test]
    fn nested_archive_builds_subdirectories() {
        let fs = build_tree(&[
            "chb06/chb06_01.edf",
            "chb06/chb06_02.edf",
            "chb07/chb07_01.edf",
        ]);
        let root = root_children(&fs);
        // Root has exactly two subdirs: chb06, chb07. No files.
        assert_eq!(root.len(), 2);
        for (n, k) in &root {
            assert_eq!(*k, FileType::Directory, "root child {n} should be dir");
        }
        let dir_names: Vec<_> = root.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(dir_names, vec!["chb06", "chb07"]);
    }

    #[test]
    fn nested_subdirs_contain_correct_files() {
        let fs = build_tree(&[
            "chb06/chb06_01.edf",
            "chb06/chb06_02.edf",
            "chb07/chb07_01.edf",
        ]);
        // Find chb06's inode.
        let root_kids = match fs.inodes.get(&ROOT_INO).unwrap() {
            INode::Dir { children } => children,
            _ => panic!(),
        };
        let chb06_ino = root_kids
            .iter()
            .find(|(_, n, _)| n == "chb06")
            .map(|(i, _, _)| *i)
            .unwrap();
        let chb06_kids = match fs.inodes.get(&chb06_ino).unwrap() {
            INode::Dir { children } => children,
            _ => panic!(),
        };
        let names: Vec<_> = chb06_kids.iter().map(|(_, n, _)| n.clone()).collect();
        assert_eq!(names, vec!["chb06_01.edf", "chb06_02.edf"]);
        // Both children must be files, not synthetic dirs.
        for (_, n, k) in chb06_kids {
            assert_eq!(*k, FileType::RegularFile, "{n} should be a file");
        }
    }

    #[test]
    fn mixed_flat_and_nested() {
        let fs = build_tree(&[
            "LICENSE.txt",
            "RECORDS",
            "PN09/PN09-1.edf",
            "PN09/PN09-2.edf",
            "PN09/Seizures-list-PN09.txt",
        ]);
        let root = root_children(&fs);
        let names: Vec<_> = root.iter().map(|(n, _)| n.clone()).collect();
        // Three at root: LICENSE.txt, PN09/, RECORDS. (Sorted.)
        assert_eq!(names, vec!["LICENSE.txt", "PN09", "RECORDS"]);
        // PN09 entry is a dir.
        let pn09_kind = root.iter().find(|(n, _)| n == "PN09").unwrap().1;
        assert_eq!(pn09_kind, FileType::Directory);
        // LICENSE.txt + RECORDS are files.
        for n in ["LICENSE.txt", "RECORDS"] {
            let k = root.iter().find(|(name, _)| name == n).unwrap().1;
            assert_eq!(k, FileType::RegularFile, "{n} should be a file");
        }
    }

    #[test]
    fn deeply_nested_paths() {
        let fs = build_tree(&["a/b/c/d/leaf.edf"]);
        // Walk a -> b -> c -> d -> leaf.edf.
        let mut cur_ino = ROOT_INO;
        for seg in ["a", "b", "c", "d", "leaf.edf"] {
            let children = match fs.inodes.get(&cur_ino).unwrap() {
                INode::Dir { children } => children,
                _ => panic!("expected dir at {cur_ino}"),
            };
            let (next_ino, _, _) =
                children.iter().find(|(_, n, _)| n == seg).expect(seg);
            cur_ino = *next_ino;
        }
        // Final inode is the file.
        let leaf = fs.inodes.get(&cur_ino).unwrap();
        assert!(matches!(leaf, INode::File { .. }));
    }

    #[test]
    fn empty_segments_are_skipped() {
        // A malformed manifest with `//` or leading `/` shouldn't
        // crash; just normalises to the non-empty segments.
        let fs = build_tree(&["/foo//bar.edf"]);
        let root = root_children(&fs);
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].0, "foo");
        assert_eq!(root[0].1, FileType::Directory);
    }

    #[test]
    fn entries_with_seizures_suffix_are_distinct_files() {
        // Regression for the chbmit-style `.edf` + `.edf.seizures`
        // sidecar pair. They share a prefix but are independent
        // files, NOT a file and its synthetic sub-entry.
        let fs = build_tree(&[
            "chb06/chb06_01.edf",
            "chb06/chb06_01.edf.seizures",
        ]);
        let root_kids = match fs.inodes.get(&ROOT_INO).unwrap() {
            INode::Dir { children } => children,
            _ => panic!(),
        };
        let chb06_ino = root_kids[0].0;
        let chb06_kids = match fs.inodes.get(&chb06_ino).unwrap() {
            INode::Dir { children } => children,
            _ => panic!(),
        };
        let names: Vec<_> = chb06_kids.iter().map(|(_, n, _)| n.clone()).collect();
        assert_eq!(names, vec!["chb06_01.edf", "chb06_01.edf.seizures"]);
        // Both are files at the same level.
        for (_, _, k) in chb06_kids {
            assert_eq!(*k, FileType::RegularFile);
        }
    }
}
