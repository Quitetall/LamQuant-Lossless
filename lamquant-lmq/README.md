# `lamquant-lmq`

Current-ABIR BCS2 neural-codec shell.

## Python backend runtime

The optional `python` feature uses a supervised helper process. It is supported
only where LamQuant can enforce complete descendant containment:

- Linux: trusted Bubblewrap at `/usr/bin/bwrap`, owned by root, executable, not
  group/world-writable, with PID namespaces available.
- Windows: a suspended child attached to a kill-on-close Job Object before
  execution resumes.

Other Unix systems fail closed. Linux packaging and CI must install Bubblewrap
and permit its user-namespace setup. Missing or untrusted containment is a
runtime error; LamQuant never falls back to an uncontained Python helper.

This boundary controls helper lifetime and prevents ordinary host writes; it is
not a confidentiality sandbox. On Linux, the helper inherits its environment
and receives a read-only view of the complete host root so Python, shared
libraries, site packages, and configured weights remain discoverable. It can
therefore read user-readable host files. Deploy only trusted helper code and
model artifacts.

The Python path is temporary. `RustBackend` becomes the portable production
path after the frozen model forward pass is available.
