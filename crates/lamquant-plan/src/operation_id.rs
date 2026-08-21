pub const EXTERNAL_OPERATION_IDS: &[&str] = &[
    "train_encoder",
    "train_snn",
    "train_tnn",
    "train_resume",
    "eagle_quick",
    "eagle_full",
    "eagle_bench",
    "eagle_lqs_l",
    "eagle_lqs_c",
    "eagle_lqs_m",
    "eagle_lqs_a",
    "eagle_perf",
    "eagle_rd",
    "eagle_h2h",
    "test_conformance",
    "test_full",
    "test_paranoid",
    "test_codec",
    "setup_pip",
    "setup_extras",
    "setup_cargo",
    "setup_musl",
    "setup_windows",
    "gui",
    "viz_lamquant-gui",
    "viz_eeglab",
    "viz_mne",
    "viz_legacy_OpenBCIGUI",
    "viz_legacy_BVAnalyzer",
    "viz_legacy_besa",
    "viz_OpenBCIGUI",
    "viz_BVAnalyzer",
    "viz_besa",
    "viz_install_lamquant_gui",
    "viz_install_mne",
    "viz_install_scope_tui",
    "viz_install_bottom",
    "viz_install_television",
    "viz_install_csvlens",
    "viz_install_gitui",
    "viz_uninstall_lamquant_gui",
    "viz_uninstall_mne",
    "viz_uninstall_scope_tui",
    "viz_uninstall_bottom",
    "viz_uninstall_television",
    "viz_uninstall_csvlens",
    "viz_uninstall_gitui",
    "cockpit_reset",
    "cockpit_checkpoints",
    "cockpit_metrics",
    "fw_list_devices",
    "fw_build_rp2350",
    "fw_build_nrf54l15",
    "fw_build_esp32p4",
    "fw_build_stm32n6",
    "fw_flash_rp2350",
    "fw_flash_nrf54l15",
    "fw_flash_esp32p4",
    "fw_flash_stm32n6",
    "fw_size_rp2350",
    "fw_size_stm32n6",
    "fw_size_esp32p4",
    "fw_size_nrf54l15",
    "fw_check_rp2350",
    "fw_check_stm32n6",
    "fw_check_esp32p4",
    "fw_check_nrf54l15",
    "fw_export",
    "fw_legacy_esp32s3",
    "cockpit_jobs",
    "cockpit_export",
    "syscheck_py",
];

pub const BLUT_OPERATION_IDS: &[&str] = &[
    "cockpit_data_prep",
    "cockpit_train_encoder",
    "cockpit_train_snn",
    "cockpit_train_oracle",
];

pub const INSTALL_OPERATION_IDS: &[&str] = &[
    "setup_install_lml",
    "setup_install_eagle",
    "setup_install_lqt",
];

pub fn install_operation_id(binary: &str) -> Option<&'static str> {
    match binary {
        "lml" => Some("setup_install_lml"),
        "eagle" => Some("setup_install_eagle"),
        "lqt" => Some("setup_install_lqt"),
        _ => None,
    }
}

pub fn is_canonical_operation_id(id: &str) -> bool {
    crate::op_spec::op_spec(id).is_some()
        || EXTERNAL_OPERATION_IDS.contains(&id)
        || BLUT_OPERATION_IDS.contains(&id)
        || INSTALL_OPERATION_IDS.contains(&id)
}

pub fn canonical_operation_ids() -> Vec<&'static str> {
    crate::op_spec::CODEC_OPERATION_IDS
        .iter()
        .chain(EXTERNAL_OPERATION_IDS)
        .chain(BLUT_OPERATION_IDS)
        .chain(INSTALL_OPERATION_IDS)
        .copied()
        .collect()
}
