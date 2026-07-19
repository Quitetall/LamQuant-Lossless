#!/usr/bin/env python3
"""Standalone stdlib-only reference decoder for construction-private DIX2 v1.

The frozen DIX1 Python oracle supplies the independently implemented temporal
predictor, native escapes, CRC32C, and binary rANS primitives. This module adds
its own DIX2 forest, TreeMED transform, packet law, selector, and canonicality
checks. It imports no Rust or LamQuant runtime package.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


def load_dix1_oracle():
    path = Path(__file__).with_name("dix1_v2_decoder.py")
    spec = importlib.util.spec_from_file_location("dix1_v2_decoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen DIX1 oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d1 = load_dix1_oracle()

MAX_PACKET_BYTES = 64 * 1024 * 1024
HEADER_LEN = 87
PACKET_CRC_OFFSET = 83
BLOCK_ROWS = 128
DIRECTORY_ENTRY_LEN = 5
MAX_CHANNELS = 64
MAX_SAMPLES = 32_768
MAX_VALUES = 131_072
MAX_LABEL_BYTES = 255
MAX_SUPPORTS = 3
MAX_SAMPLE_RATE_MHZ = 4_000_000


class DecodeError(d1.DecodeError):
    """The packet violates the construction DIX2 v1 contract."""


@dataclass
class ChannelModel:
    nonzero: d1.Counts
    sign: d1.Counts
    exponent: list[d1.Counts]
    mantissa: list[d1.Counts]

    @classmethod
    def build(cls, bit_depth: int) -> "ChannelModel":
        return cls(
            d1.Counts(),
            d1.Counts(),
            [d1.Counts() for _ in range(bit_depth)],
            [d1.Counts() for _ in range(bit_depth)],
        )


class EntropyDecoder:
    def __init__(self, payload: bytes, channels: int, values: int, bit_depth: int):
        if not 1 <= bit_depth <= 33:
            raise DecodeError("DIX2 entropy bit depth is outside bounds")
        self.bit_depth = bit_depth
        self.coder = d1.BinaryRansDecoder(payload, values * (2 * bit_depth + 1))
        self.models = [ChannelModel.build(bit_depth) for _ in range(channels)]

    def read_context(self, context: d1.Counts) -> int:
        bit = self.coder.read(context.probability_one())
        context.observe(bit)
        return bit

    def read_value(self, channel: int) -> int:
        model = self.models[channel]
        if self.read_context(model.nonzero) == 0:
            return 0
        exponent = 0
        while self.read_context(model.exponent[exponent]) == 1:
            exponent += 1
            if exponent >= self.bit_depth:
                raise DecodeError("DIX2 residual exponent exceeds bit depth")
        sign = self.read_context(model.sign)
        magnitude = 1 << exponent
        for position in reversed(range(exponent)):
            magnitude |= self.read_context(model.mantissa[position]) << position
        return magnitude if sign == 0 else -magnitude


class EntropyEncoder:
    def __init__(self, channels: int, values: int, bit_depth: int):
        if not 1 <= bit_depth <= 33:
            raise DecodeError("DIX2 entropy bit depth is outside bounds")
        self.bit_depth = bit_depth
        self.coder = d1.BinaryRansEncoder(values * (2 * bit_depth + 1))
        self.models = [ChannelModel.build(bit_depth) for _ in range(channels)]

    def push_context(self, context: d1.Counts, bit: int) -> None:
        self.coder.push(bit, context.probability_one())
        context.observe(bit)

    def push_value(self, channel: int, value: int) -> None:
        model = self.models[channel]
        magnitude = abs(value)
        self.push_context(model.nonzero, int(magnitude != 0))
        if magnitude == 0:
            return
        exponent = magnitude.bit_length() - 1
        if exponent >= self.bit_depth:
            raise DecodeError("DIX2 residual magnitude exceeds bit depth")
        for position in range(exponent + 1):
            self.push_context(model.exponent[position], int(position < exponent))
        self.push_context(model.sign, int(value < 0))
        for position in reversed(range(exponent)):
            self.push_context(model.mantissa[position], (magnitude >> position) & 1)


def derive_tree_topology(
    identities: list[d1.Identity],
) -> list[list[tuple[int, int]]]:
    parsed = sorted(
        (d1.parse_channel(identity, index) for index, identity in enumerate(identities)),
        key=d1.ParsedChannel.sort_key,
    )
    if [channel.presented_index for channel in parsed] != list(range(len(identities))):
        raise DecodeError("DIX2 identity section is not exact canonical order")
    topology: list[list[tuple[int, int]]] = []
    for channel_index, channel in enumerate(parsed):
        non_reference: list[tuple[int, int]] = []
        references: list[tuple[int, int]] = []
        if channel.partition == 0:
            for prior_index in reversed(range(channel_index)):
                prior = parsed[prior_index]
                if prior.partition != 0:
                    continue
                shared = shared_endpoints(channel, prior)
                selected = next(
                    (item for item in shared if not item[0].startswith("R:")),
                    shared[0] if shared else None,
                )
                if selected is None:
                    continue
                endpoint, coefficient = selected
                support = (prior_index, coefficient)
                if endpoint.startswith("R:"):
                    references.append(support)
                else:
                    non_reference.append(support)
        topology.append((non_reference + references)[:MAX_SUPPORTS])
    return topology


def shared_endpoints(current, prior) -> list[tuple[str, int]]:
    current_endpoints = (
        (current.positive_endpoint, 1),
        (current.negative_endpoint, -1),
    )
    prior_endpoints = (
        (prior.positive_endpoint, 1),
        (prior.negative_endpoint, -1),
    )
    shared = [
        (endpoint, current_sign * prior_sign)
        for endpoint, current_sign in current_endpoints
        for prior_endpoint, prior_sign in prior_endpoints
        if endpoint and endpoint == prior_endpoint
    ]
    return sorted(shared, key=lambda item: (item[0], item[1]))


def tree_prediction(
    supports: list[tuple[int, int]], innovations: list[int]
) -> int:
    if not supports:
        return 0
    predictions = []
    for parent, coefficient in supports:
        if parent >= len(innovations) or coefficient not in (-1, 1):
            raise DecodeError("DIX2 TreeMED support is unavailable")
        predictions.append(
            d1.checked_i64(
                coefficient * innovations[parent], "DIX2 support innovation"
            )
        )
    predictions.sort()
    return predictions[(len(predictions) - 1) // 2]


def tree_forward(
    innovations: list[int], topology: list[list[tuple[int, int]]]
) -> list[int]:
    if len(innovations) != len(topology):
        raise DecodeError("DIX2 innovation row has the wrong channel count")
    return [
        d1.checked_i64(
            innovation - tree_prediction(topology[channel], innovations),
            "DIX2 tree residual",
        )
        for channel, innovation in enumerate(innovations)
    ]


def tree_inverse(
    residuals: list[int], topology: list[list[tuple[int, int]]]
) -> list[int]:
    if len(residuals) != len(topology):
        raise DecodeError("DIX2 residual row has the wrong channel count")
    innovations: list[int] = []
    for channel, residual in enumerate(residuals):
        innovations.append(
            d1.checked_i64(
                residual + tree_prediction(topology[channel], innovations),
                "DIX2 inverse innovation",
            )
        )
    return innovations


def permitted(profile: int, mode: int) -> bool:
    return {
        0: mode in range(4),
        1: mode in (0, 1),
        2: mode == 0,
        3: mode == 1,
        4: mode == 2,
        5: mode == 3,
    }.get(profile, False)


def canonical_block_payload(
    profile: int,
    stable_signal: list[list[int]],
    identities: list[d1.Identity],
    block_start_session: d1.Dix1Session,
    topology: list[list[tuple[int, int]]],
    bit_depth: int,
) -> tuple[int, bytes]:
    rows = len(stable_signal[0])
    channels = len(stable_signal)
    raw = d1.encode_raw(stable_signal)
    delta = d1.encode_delta(stable_signal)
    temporal_session = copy.deepcopy(block_start_session)
    temporal_entropy = EntropyEncoder(channels, channels * rows, bit_depth)
    tree_entropy = EntropyEncoder(channels, channels * rows, bit_depth + 1)
    for row in range(rows):
        canonical = [stable_signal[identity.stable_id][row] for identity in identities]
        innovations = temporal_session.forward_row(canonical)
        residuals = tree_forward(innovations, topology)
        for channel in range(channels):
            temporal_entropy.push_value(channel, innovations[channel])
            tree_entropy.push_value(channel, residuals[channel])
    temporal = temporal_entropy.coder.finish()
    tree = tree_entropy.coder.finish()
    candidates = {
        0: ((0, raw), (1, delta), (2, temporal), (3, tree)),
        1: ((0, raw), (1, delta)),
        2: ((0, raw),),
        3: ((1, delta),),
        4: ((2, temporal),),
        5: ((3, tree),),
    }.get(profile)
    if candidates is None:
        raise DecodeError("DIX2 carrier profile is invalid")
    return min(candidates, key=lambda candidate: (len(candidate[1]), candidate[0]))


def decode_packet(packet: bytes) -> dict[str, object]:
    if not HEADER_LEN <= len(packet) <= MAX_PACKET_BYTES:
        raise DecodeError("DIX2 packet length is outside bounds")
    if packet[:7] != b"LMO1\x03\x00\x03" or packet[7:11] != b"DIX2":
        raise DecodeError("DIX2 envelope or magic is invalid")
    if packet[11] != 1 or packet[12] != 1:
        raise DecodeError("DIX2 body version or flags are invalid")

    bit_depth = packet[13]
    profile = packet[14]
    channels = d1.u16le(packet, 15)
    tile_count = d1.u16le(packet, 17)
    samples = d1.u32le(packet, 19)
    sample_rate_mhz = d1.u32le(packet, 23)
    model_id = d1.u32le(packet, 27)
    identity_len = d1.u32le(packet, 63)
    topology_len = d1.u32le(packet, 67)
    directory_len = d1.u32le(packet, 71)
    payload_len = d1.u32le(packet, 75)
    decoded_crc = d1.u32le(packet, 79)
    stored_packet_crc = d1.u32le(packet, 83)
    values = channels * samples
    if not (
        1 <= channels <= MAX_CHANNELS
        and 1 <= samples <= MAX_SAMPLES
        and values <= MAX_VALUES
        and 1 <= bit_depth <= 32
        and 1 <= sample_rate_mhz <= MAX_SAMPLE_RATE_MHZ
    ):
        raise DecodeError("DIX2 dimensions are outside bounds")
    blocks = (samples + BLOCK_ROWS - 1) // BLOCK_ROWS
    if (
        profile not in range(6)
        or tile_count != blocks
        or model_id != 0
        or packet[31:63] != bytes(32)
        or directory_len != blocks * DIRECTORY_ENTRY_LEN
        or not channels * 4 <= identity_len <= channels * (3 + MAX_LABEL_BYTES)
        or not channels <= topology_len <= channels * (1 + 2 * MAX_SUPPORTS)
    ):
        raise DecodeError("DIX2 construction header is invalid")

    identity_start = HEADER_LEN
    identity_end = identity_start + identity_len
    topology_end = identity_end + topology_len
    directory_end = topology_end + directory_len
    packet_end = directory_end + payload_len
    if packet_end != len(packet):
        raise DecodeError("DIX2 section lengths do not cover packet")
    if stored_packet_crc != d1.packet_crc32c(packet):
        raise DecodeError("DIX2 packet CRC32C mismatch")

    identities = d1.parse_identities(packet[identity_start:identity_end], channels)
    topology = d1.parse_topology(packet[identity_end:topology_end], channels)
    derived = derive_tree_topology(identities)
    if topology != derived:
        raise DecodeError("DIX2 topology does not match its deterministic forest")
    directory = packet[topology_end:directory_end]
    payload = packet[directory_end:]
    entries: list[tuple[int, int]] = []
    payload_sum = 0
    for entry in range(blocks):
        offset = entry * DIRECTORY_ENTRY_LEN
        mode = directory[offset]
        length = d1.u32le(directory, offset + 1)
        if mode not in range(4) or not permitted(profile, mode):
            raise DecodeError("DIX2 block mode is not permitted by profile")
        payload_sum += length
        if payload_sum > len(payload):
            raise DecodeError("DIX2 block payload lengths exceed payload section")
        entries.append((mode, length))
    if payload_sum != len(payload):
        raise DecodeError("DIX2 directory does not cover payload exactly")

    # Disabled incidence still retains topology-conditioned temporal priors in
    # the frozen DIX1 state machine.  Supplying empty supports changes those
    # priors and eventually diverges on mixed referential/bipolar montages.
    temporal = d1.Dix1Session(
        d1.derive_topology(identities), bit_depth, sample_rate_mhz, False
    )
    stable_signal: list[list[int]] = [[] for _ in range(channels)]
    tile_modes: list[int] = []
    payload_offset = 0
    event_count = 0
    for block, (mode, length) in enumerate(entries):
        rows = min(BLOCK_ROWS, samples - block * BLOCK_ROWS)
        end = payload_offset + length
        block_payload = payload[payload_offset:end]
        block_start_session = copy.deepcopy(temporal)
        if mode == 0:
            stable_block = d1.decode_raw(block_payload, channels, rows)
            advance_escape_block(temporal, identities, stable_block, rows)
        elif mode == 1:
            stable_block = d1.decode_delta(block_payload, channels, rows)
            advance_escape_block(temporal, identities, stable_block, rows)
        else:
            entropy_bit_depth = bit_depth if mode == 2 else bit_depth + 1
            entropy = EntropyDecoder(
                block_payload, channels, channels * rows, entropy_bit_depth
            )
            stable_block = [[] for _ in range(channels)]
            for _ in range(rows):
                coded = [entropy.read_value(channel) for channel in range(channels)]
                innovations = tree_inverse(coded, topology) if mode == 3 else coded
                canonical = temporal.inverse_row(innovations)
                for channel, sample in enumerate(canonical):
                    stable_block[identities[channel].stable_id].append(sample)
            entropy.coder.finish()
            event_count += entropy.coder.event_count
        canonical_mode, canonical_payload = canonical_block_payload(
            profile,
            stable_block,
            identities,
            block_start_session,
            topology,
            bit_depth,
        )
        if mode != canonical_mode or block_payload != canonical_payload:
            raise DecodeError("DIX2 block is not byte-canonical for its profile")
        for stable_id in range(channels):
            stable_signal[stable_id].extend(stable_block[stable_id])
        tile_modes.append(mode)
        payload_offset = end

    if payload_offset != len(payload) or any(
        len(channel) != samples for channel in stable_signal
    ):
        raise DecodeError("DIX2 block decode did not cover the window")
    decoded_bytes = bytearray()
    sample_min = -(1 << (bit_depth - 1))
    sample_max = (1 << (bit_depth - 1)) - 1
    for channel in stable_signal:
        for sample in channel:
            if not sample_min <= sample <= sample_max:
                raise DecodeError("DIX2 decoded sample exceeds declared bit depth")
            if not -(1 << 31) <= sample < 1 << 31:
                raise DecodeError("DIX2 decoded sample exceeds i32")
            decoded_bytes.extend(struct.pack("<i", sample))
    if d1.crc32c(decoded_bytes) != decoded_crc:
        raise DecodeError("DIX2 decoded-sample CRC32C mismatch")
    stable_identities = sorted(identities, key=lambda identity: identity.stable_id)
    return {
        "sample_rate_mhz": sample_rate_mhz,
        "bit_depth": bit_depth,
        "stable_ids": [identity.stable_id for identity in stable_identities],
        "labels": [identity.label for identity in stable_identities],
        "tile_modes": tile_modes,
        "event_count": event_count,
        "samples": stable_signal,
    }


def advance_escape_block(
    temporal: d1.Dix1Session,
    identities: list[d1.Identity],
    stable_block: list[list[int]],
    rows: int,
) -> None:
    for row in range(rows):
        canonical = [stable_block[identity.stable_id][row] for identity in identities]
        temporal.advance_row(canonical)


def main() -> int:
    packet = sys.stdin.buffer.read(MAX_PACKET_BYTES + 1)
    if len(packet) > MAX_PACKET_BYTES:
        raise DecodeError("DIX2 packet exceeds 64 MiB")
    result = decode_packet(packet)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (d1.DecodeError, MemoryError, OverflowError, struct.error) as error:
        print(f"DIX2 reference decode failed: {error}", file=sys.stderr)
        raise SystemExit(2)
