use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn command_stdout(program: &str, args: &[&str], cwd: Option<&Path>) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn rustc_field(verbose: &str, prefix: &str) -> String {
    verbose
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .unwrap_or("unknown")
        .to_owned()
}

fn emit_env(name: &str, value: &str) {
    assert!(
        !value.contains(['\n', '\r']),
        "invalid build provenance value"
    );
    println!("cargo:rustc-env={name}={value}");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_BENCHMARK_CLI").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repository = manifest_dir
        .parent()
        .expect("lamquant-lml-optimum must live under codec-lossless");

    // These are the source inputs linked into the benchmark executable. Git's
    // HEAD/index watches also refresh identity when a commit or staging state
    // changes. The recorded dirty bit is the whole codec-lossless worktree.
    for path in [
        manifest_dir.join("Cargo.toml"),
        manifest_dir.join("src"),
        repository.join("Cargo.toml"),
        repository.join("Cargo.lock"),
        repository.join("lamquant-lml-mcu/Cargo.toml"),
        repository.join("lamquant-lml-mcu/src"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for git_path in ["HEAD", "index"] {
        if let Some(path) = command_stdout(
            "git",
            &["rev-parse", "--git-path", git_path],
            Some(repository),
        ) {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let git_head = command_stdout("git", &["rev-parse", "HEAD"], Some(repository))
        .unwrap_or_else(|| "unknown".to_owned());
    let git_dirty = command_stdout(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
        Some(repository),
    )
    .map_or(
        "unknown",
        |status| if status.is_empty() { "false" } else { "true" },
    );

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_verbose = command_stdout(&rustc, &["--version", "--verbose"], None)
        .unwrap_or_else(|| "release: unknown\ncommit-hash: unknown".to_owned());
    let rustc_version =
        command_stdout(&rustc, &["--version"], None).unwrap_or_else(|| "rustc unknown".to_owned());
    let rustc_commit = rustc_field(&rustc_verbose, "commit-hash: ");

    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort_unstable();

    emit_env("LQ_OPTIMUM_GIT_HEAD", &git_head);
    emit_env("LQ_OPTIMUM_GIT_DIRTY", git_dirty);
    emit_env(
        "LQ_OPTIMUM_BUILD_PROFILE",
        &env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit_env(
        "LQ_OPTIMUM_BUILD_TARGET",
        &env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit_env("LQ_OPTIMUM_BUILD_FEATURES", &features.join(","));
    emit_env("LQ_OPTIMUM_RUSTC_VERSION", &rustc_version);
    emit_env("LQ_OPTIMUM_RUSTC_COMMIT", &rustc_commit);
}
