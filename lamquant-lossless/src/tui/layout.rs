//! Layout engine — defines how panels are arranged on screen.
//!
//! Layouts are data-driven: define splits as percentages or fixed sizes,
//! and the engine computes Rects for each panel slot.

use ratatui::prelude::*;

/// A layout slot — either a single panel or a nested split.
#[derive(Debug, Clone)]
pub enum Slot {
    /// Single panel fills this slot.
    Panel { id: String },
    /// Vertical split: top and bottom.
    VSplit {
        top: Box<Slot>,
        bottom: Box<Slot>,
        ratio: u16,
    },
    /// Horizontal split: left and right.
    HSplit {
        left: Box<Slot>,
        right: Box<Slot>,
        ratio: u16,
    },
}

/// Screen layout definition.
#[derive(Debug, Clone)]
pub struct ScreenLayout {
    pub root: Slot,
}

impl ScreenLayout {
    /// Single panel fills entire screen.
    pub fn full(panel_id: &str) -> Self {
        Self {
            root: Slot::Panel {
                id: panel_id.to_string(),
            },
        }
    }

    /// Left panel + right panel (horizontal split).
    pub fn hsplit(left: &str, right: &str, ratio: u16) -> Self {
        Self {
            root: Slot::HSplit {
                left: Box::new(Slot::Panel {
                    id: left.to_string(),
                }),
                right: Box::new(Slot::Panel {
                    id: right.to_string(),
                }),
                ratio,
            },
        }
    }

    /// Top panel + bottom panel (vertical split).
    pub fn vsplit(top: &str, bottom: &str, ratio: u16) -> Self {
        Self {
            root: Slot::VSplit {
                top: Box::new(Slot::Panel {
                    id: top.to_string(),
                }),
                bottom: Box::new(Slot::Panel {
                    id: bottom.to_string(),
                }),
                ratio,
            },
        }
    }

    /// Compute rects for all panel IDs given a total area.
    pub fn compute_rects(&self, area: Rect) -> Vec<(String, Rect)> {
        let mut result = Vec::new();
        Self::compute_slot(&self.root, area, &mut result);
        result
    }

    fn compute_slot(slot: &Slot, area: Rect, out: &mut Vec<(String, Rect)>) {
        match slot {
            Slot::Panel { id } => {
                out.push((id.clone(), area));
            }
            Slot::VSplit { top, bottom, ratio } => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(*ratio),
                        Constraint::Percentage(100 - ratio),
                    ])
                    .split(area);
                Self::compute_slot(top, chunks[0], out);
                Self::compute_slot(bottom, chunks[1], out);
            }
            Slot::HSplit { left, right, ratio } => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(*ratio),
                        Constraint::Percentage(100 - ratio),
                    ])
                    .split(area);
                Self::compute_slot(left, chunks[0], out);
                Self::compute_slot(right, chunks[1], out);
            }
        }
    }
}
