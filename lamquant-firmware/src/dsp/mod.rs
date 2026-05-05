//! DSP front-end: prefilter → LPC → lifting DWT → WHT.
//!
//! Shared by both Mode 1 (Neural) and Mode 2 (Lossless). Operates on the
//! 21×2500 Q31 ADC buffer in-place. Output: 21×313 L3 approximation +
//! detail subbands (D1=1250, D2=625, D3=312 samples).

pub mod biquad;
pub mod lifting;
pub mod lpc;
pub mod wht;
