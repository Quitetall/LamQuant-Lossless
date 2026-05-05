//! Quality mode + activity level — cross-cutting enums used by the SNN
//! decision logic, FSQ adaptive level selection, and detail thresholding.

/// LQS quality mode. Selected by host serial command (`0` / `1` / `2`)
/// or by SNN auto-mode dispatcher.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[repr(u8)]
pub enum QualityMode {
    /// 1-3 ch L3 only, ~525:1 CR. Battery-critical / quiescent monitoring.
    Alerting = 0,
    /// L3 + L2 detail. ~80-122:1 CR. Standard ambulatory.
    Monitoring = 1,
    /// All subbands, gentle threshold. ~63:1 CR. Diagnostic recording.
    Clinical = 2,
}

impl QualityMode {
    pub const COUNT: usize = 3;

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Alerting),
            1 => Some(Self::Monitoring),
            2 => Some(Self::Clinical),
            _ => None,
        }
    }
}

/// Per-group SNN classification. Drives FSQ level + detail threshold.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[repr(u8)]
pub enum ActivityLevel {
    Quiescent = 0,
    Active = 1,
    High = 2,
}

impl ActivityLevel {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Quiescent,
            1 => Self::Active,
            _ => Self::High,
        }
    }
}
