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

The Python path is temporary. `RustBackend` becomes the portable production
path after the frozen model forward pass is available.
