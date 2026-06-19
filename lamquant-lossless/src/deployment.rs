use crate::lpc::LpcMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LosslessMode {
    Mcu,
    Basestation,
}

impl LosslessMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mcu => "mcu",
            Self::Basestation => "basestation",
        }
    }

    pub const fn default_lpc_mode(self) -> LpcMode {
        match self {
            Self::Mcu => LpcMode::Fixed,
            Self::Basestation => LpcMode::Adaptive { max_order: 16 },
        }
    }
}
