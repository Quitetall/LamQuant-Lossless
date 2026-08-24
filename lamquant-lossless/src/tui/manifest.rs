// SPDX-License-Identifier: AGPL-3.0-or-later
//! The codec's own declaration of the shell it wants (ADR 0053).
//!
//! Until now this did not exist. `lml` called the framework's bare
//! `run_interactive()`, which resolved to `ShellProfile::Spoke`, and
//! `lamquant-tui` held the definition of a front-end it should not know about:
//! the build label `"lml"`, the Codec Hub tile, the four utility tiles, and the
//! home screen. A shared TUI crate carrying a specific product's tile table is
//! exactly the coupling ADR 0053 exists to remove, and it is the mirror of what
//! the hub fixed when it grew `tui/src/tui/manifest.rs`.
//!
//! Now the codec states it. `lamquant-tui` still supplies the shell -- event
//! loop, router, layout, widgets, terminal lifecycle, panic recovery -- and
//! this file supplies the product.
//!
//! The tables below are a VERBATIM copy of what `ShellManifest::spoke()` held,
//! so this move changes ownership and nothing else. The one field that is
//! wrong on purpose is `home_screen`; see its comment.

use lamquant_tui::{PanelRegistration, ShellManifest, TileSpec};

/// The single workflow the codec offers. `lml` is its own binary, so the tile
/// is install-gated on a binary that is by definition present -- it renders
/// normally and routes to the codec hub.
const WORKFLOWS: &[TileSpec] = &[TileSpec::new(
    "1",
    "Codec Hub",
    "Compress · decompress · browse · verify",
    "lml",
    lamquant_tui::router::SCREEN_CODEC_HUB,
)];

/// Utility tiles beneath the workflow. Identical to the hub's, because these
/// four screens are framework-provided and every front-end that has them wants
/// the same four. They are duplicated rather than shared: a shared constant
/// would put the tile table back in the framework, which is the thing being
/// undone here.
const UTILITIES: &[TileSpec] = &[
    TileSpec::always(
        "N",
        "Peers",
        "Remote LamQuant devices · SSH targets",
        lamquant_tui::router::SCREEN_PEERS,
    ),
    TileSpec::always(
        "s",
        "Settings",
        "Workers · paths · device profiles",
        lamquant_tui::router::SCREEN_SETTINGS,
    ),
    TileSpec::always(
        "i",
        "Install & setup",
        "Wizard · dependencies · syscheck · GPU probe",
        lamquant_tui::router::SCREEN_SETUP,
    ),
    TileSpec::always(
        "t",
        "Diagnostics",
        "Internal Testing Suite · Crashlog Viewer · Health Check",
        lamquant_tui::router::SCREEN_TEST,
    ),
];

/// The codec declares no panels of its own: every screen it reaches is
/// framework-provided. Empty is the honest value -- a front-end that cannot
/// service a screen says nothing rather than registering a dead tile.
const PANELS: &[PanelRegistration] = &[];

/// What `lml` asks the shared shell for.
pub const fn lml() -> ShellManifest {
    ShellManifest {
        build_label: "lml",
        // NOT `SCREEN_CODEC_HUB`, yet. ADR 0053's Context section names this as
        // the noncompliance: "Running `lml` with no args boots into a mini-hub
        // ... instead of going directly to lossless codec operations." It is
        // carried over unchanged so that THIS commit is provably inert -- the
        // move of ownership and the change of behaviour are separate acts, and
        // conflating them would make it impossible to show the move changed
        // nothing. Correcting it is now a one-line edit in this file, which is
        // the whole point of the field living here.
        home_screen: lamquant_tui::router::SCREEN_MAIN,
        workflow_tiles: WORKFLOWS,
        utility_tiles: UTILITIES,
        hub_settings: false,
        panels: PANELS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codec must declare exactly what the framework used to hold for it.
    /// This is what makes "ownership moved, behaviour did not" checkable rather
    /// than merely asserted in a commit message.
    #[test]
    fn the_codec_declares_what_the_spoke_profile_held() {
        let m = lml();
        assert_eq!(m.build_label, "lml");
        assert!(!m.hub_settings);
        assert_eq!(m.home_screen, lamquant_tui::router::SCREEN_MAIN);

        let keys: Vec<&str> = m.workflow_tiles.iter().map(|t| t.key).collect();
        assert_eq!(keys, ["1"]);
        let bins: Vec<&str> = m.workflow_tiles.iter().map(|t| t.binary).collect();
        assert_eq!(bins, ["lml"]);

        let util: Vec<&str> = m.utility_tiles.iter().map(|t| t.key).collect();
        assert_eq!(util, ["N", "s", "i", "t"]);
    }

    /// Byte-for-byte against the framework's copy, field by field. When the
    /// framework finally drops `ShellManifest::spoke()` this test goes with it,
    /// and until then it is the only thing that would catch the two drifting
    /// apart -- which is the failure that would make the eventual deletion
    /// silently change what `lml` renders.
    #[test]
    fn it_matches_the_framework_copy_it_replaces() {
        let mine = lml();
        let theirs = ShellManifest::spoke();
        assert_eq!(mine.build_label, theirs.build_label);
        assert_eq!(mine.home_screen, theirs.home_screen);
        assert_eq!(mine.hub_settings, theirs.hub_settings);
        assert_eq!(mine.panels.len(), theirs.panels.len());

        let f = |t: &&TileSpec| (t.key, t.label, t.description, t.binary, t.screen);
        let a: Vec<_> = mine.workflow_tiles.iter().collect();
        let b: Vec<_> = theirs.workflow_tiles.iter().collect();
        assert_eq!(
            a.iter().map(f).collect::<Vec<_>>(),
            b.iter().map(f).collect::<Vec<_>>()
        );
        let a: Vec<_> = mine.utility_tiles.iter().collect();
        let b: Vec<_> = theirs.utility_tiles.iter().collect();
        assert_eq!(
            a.iter().map(f).collect::<Vec<_>>(),
            b.iter().map(f).collect::<Vec<_>>()
        );
    }
}
