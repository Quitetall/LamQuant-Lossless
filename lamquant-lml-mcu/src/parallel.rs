//! Host-only Rayon channel execution adapter.
//!
//! Packet policy stays in [`crate::lml`]. This module only maps independent
//! channel work while preserving input order.

use alloc::vec::Vec;

use rayon::prelude::*;

use crate::error::LmlResult;
use crate::lml::{
    encode_channel_with_policy, synthesize_channel_signal, ChannelEncodeOutput, EncodePolicy,
};
use crate::lpc::LpcMode;

pub(crate) fn encode_channels(
    channels: &[&[i64]],
    n_levels: u8,
    mode: LpcMode,
    policy: EncodePolicy,
    per_channel_limit: usize,
) -> LmlResult<Vec<ChannelEncodeOutput>> {
    channels
        .par_iter()
        .map(|&channel| {
            encode_channel_with_policy(channel, n_levels, mode, policy, per_channel_limit)
        })
        .collect()
}

pub(crate) fn synthesize_channels(
    channels: Vec<Vec<(Vec<i32>, Vec<i64>)>>,
    n_levels: u8,
) -> LmlResult<Vec<Vec<i64>>> {
    channels
        .into_par_iter()
        .map(|subbands| synthesize_channel_signal(subbands, n_levels))
        .collect()
}
