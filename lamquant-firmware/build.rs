//! Place memory.x in OUT_DIR so the linker (-Tmemory.x) finds our memory layout.
//!
//! memory.x defines MEMORY + REGION_ALIAS, then INCLUDEs riscv-rt's link.x.
//! Named `memory.x` (not `device.x`) to avoid collision with rp235x-pac's
//! IRQ-vector device.x.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out.join("memory.x"), include_bytes!("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());

    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
