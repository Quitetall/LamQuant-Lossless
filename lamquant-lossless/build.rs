// Capture the git commit short SHA at build time so the TUI splash can
// echo `commit=` next to the version. Mirrors the Python TUI's `git_commit`
// helper. Falls back to "dev" if git is unavailable (sdist tarballs,
// reproducible builds, vendored builds).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");

    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "dev".to_string());

    println!("cargo:rustc-env=LAMQUANT_GIT_COMMIT={}", commit);
}
