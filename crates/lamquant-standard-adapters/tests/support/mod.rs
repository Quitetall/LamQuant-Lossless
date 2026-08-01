// SPDX-License-Identifier: AGPL-3.0-or-later

use abir_adapter::ForeignObject;
use std::path::{Component, Path, PathBuf};

pub fn dump_package33_output(label: &str, object: &ForeignObject) {
    let Some(root) = std::env::var_os("LAMQUANT_PACKAGE33_OUTPUT_DIR") else {
        return;
    };
    assert!(
        !label.is_empty()
            && Path::new(label)
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "Package 33 output label must be one safe path component"
    );
    let target = PathBuf::from(root).join(label);
    if target.exists() {
        std::fs::remove_dir_all(&target).expect("remove stale Package 33 output tree");
    }
    std::fs::create_dir_all(&target).expect("create Package 33 output tree");
    for entry in &object.entries {
        let relative = Path::new(&entry.path);
        assert!(
            !relative.as_os_str().is_empty()
                && relative
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "adapter emitted unsafe Package 33 path: {}",
            entry.path
        );
        let destination = target.join(relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).expect("create Package 33 member parent");
        }
        std::fs::write(destination, &entry.bytes).expect("write Package 33 output member");
    }
}
