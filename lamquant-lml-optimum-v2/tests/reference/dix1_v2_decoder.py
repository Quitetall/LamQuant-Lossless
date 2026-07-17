#!/usr/bin/env python3
"""Standalone stdlib-only reference decoder for construction-private DIX1 v2.

This intentionally imports no LamQuant package.  It is a cross-language oracle
for packet framing, block modes, KT/rANS residuals, and the fixed-integer DIX1
predictor.  The Rust integration test supplies one complete packet on stdin and
compares this decoder's JSON result with the original signal.
"""

from __future__ import annotations

import copy
import json
import struct
import sys
from array import array
from dataclasses import dataclass


MAX_PACKET_BYTES = 64 * 1024 * 1024
HEADER_LEN = 87
PACKET_CRC_OFFSET = 83
BLOCK_ROWS = 128
DIRECTORY_ENTRY_LEN = 5
MAX_CHANNELS = 64
MAX_SAMPLES = 32_768
MAX_VALUES = 131_072
MAX_LABEL_BYTES = 255
MAX_SUPPORTS = 4
MAX_SAMPLE_RATE_MHZ = 4_000_000
REFERENCE_TOKENS = ("REF", "LE", "AR", "AVG", "CZREF")
AUX_PREFIXES = (
    "ECG",
    "EKG",
    "RESP",
    "SPO2",
    "SP02",
    "PULSE",
    "HEART",
    "HR",
    "EMG",
    "EOG",
    "MARK",
    "MK",
    "TRIGGER",
    "EVENT",
    "TEMP",
    "CO2",
    "AIRFLOW",
)
ELECTRODE_FAMILIES = (
    "FP",
    "AF",
    "FC",
    "FT",
    "TP",
    "CP",
    "PO",
    "F",
    "T",
    "C",
    "P",
    "O",
    "I",
    "A",
    "M",
)

CDF_BITS = 15
CDF_TOTAL = 1 << CDF_BITS
RANS_L = 1 << 23

TEMPORAL_LAG_MS = (1, 4, 16, 64)
SCORE_Q = 256
SCORE_DECAY_SHIFT = 5
MIX_WEIGHT_NUMERATOR = 1 << 20
INCIDENCE_PRIOR_COST = 64 * SCORE_Q

WEIGHT_Q = 20
WEIGHT_ONE = 1 << WEIGHT_Q
WEIGHT_LIMIT = 16 * WEIGHT_ONE
COVARIANCE_ONE = 1 << 36
COVARIANCE_LIMIT = (1 << 42) - 1
GAIN_ONE = 1 << 40
GAIN_LIMIT = 8 * GAIN_ONE
FORGETTING_NUMERATOR = 253
FORGETTING_DENOMINATOR = 256

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
I128_MIN = -(1 << 127)
I128_MAX = (1 << 127) - 1
U64_MAX = (1 << 64) - 1


class DecodeError(ValueError):
    """The packet violates the construction DIX1 v2 contract."""


def checked_i128(value: int, label: str) -> int:
    if not I128_MIN <= value <= I128_MAX:
        raise DecodeError(f"{label} exceeds signed i128")
    return value


def checked_i64(value: int, label: str) -> int:
    if not I64_MIN <= value <= I64_MAX:
        raise DecodeError(f"{label} exceeds signed i64")
    return value


def round_ratio_away(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise DecodeError("fixed ratio denominator must be positive")
    magnitude = (abs(numerator) + denominator // 2) // denominator
    return magnitude if numerator >= 0 else -magnitude


def crc32c(data: bytes | bytearray) -> int:
    state = 0xFFFFFFFF
    for byte in data:
        state ^= byte
        for _ in range(8):
            state = (state >> 1) ^ (0x82F63B78 if state & 1 else 0)
    return state ^ 0xFFFFFFFF


def packet_crc32c(packet: bytes) -> int:
    zeroed = bytearray(packet)
    zeroed[PACKET_CRC_OFFSET : PACKET_CRC_OFFSET + 4] = b"\0\0\0\0"
    return crc32c(zeroed)


def u16le(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise DecodeError("truncated u16")
    return int.from_bytes(data[offset : offset + 2], "little")


def u32le(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise DecodeError("truncated u32")
    return int.from_bytes(data[offset : offset + 4], "little")


@dataclass(frozen=True)
class Identity:
    stable_id: int
    label: str


def parse_identities(data: bytes, channels: int) -> list[Identity]:
    identities: list[Identity] = []
    offset = 0
    for _ in range(channels):
        stable_id = u16le(data, offset)
        offset += 2
        if offset >= len(data):
            raise DecodeError("identity label length is truncated")
        label_len = data[offset]
        offset += 1
        end = offset + label_len
        if label_len == 0 or end > len(data):
            raise DecodeError("identity label is empty or truncated")
        encoded = data[offset:end]
        if any(byte < 0x20 or byte > 0x7E for byte in encoded):
            raise DecodeError("identity label is not printable ASCII")
        identities.append(Identity(stable_id, encoded.decode("ascii")))
        offset = end
    if offset != len(data):
        raise DecodeError("identity section has trailing bytes")
    if sorted(identity.stable_id for identity in identities) != list(range(channels)):
        raise DecodeError("stable IDs are not a contiguous permutation")
    return identities


def parse_topology(data: bytes, channels: int) -> list[list[tuple[int, int]]]:
    topology: list[list[tuple[int, int]]] = []
    offset = 0
    for channel in range(channels):
        if offset >= len(data):
            raise DecodeError("topology support count is truncated")
        count = data[offset]
        offset += 1
        if count > MAX_SUPPORTS:
            raise DecodeError("topology support count exceeds four")
        supports: list[tuple[int, int]] = []
        for _ in range(count):
            if offset + 2 > len(data):
                raise DecodeError("topology support is truncated")
            prior = data[offset]
            coefficient = data[offset + 1]
            coefficient = coefficient - 256 if coefficient >= 128 else coefficient
            offset += 2
            if prior >= channel or coefficient not in (-1, 1):
                raise DecodeError("topology support is not causal signed incidence")
            supports.append((prior, coefficient))
        topology.append(supports)
    if offset != len(data):
        raise DecodeError("topology section has trailing bytes")
    return topology


@dataclass(frozen=True)
class ParsedChannel:
    presented_index: int
    stable_id: int
    normalized_label: str
    partition: int
    kind: int
    positive_endpoint: str
    negative_endpoint: str

    def sort_key(self) -> tuple[int, int, str, str, str, int]:
        return (
            self.partition,
            self.kind,
            self.positive_endpoint,
            self.negative_endpoint,
            self.normalized_label,
            self.stable_id,
        )


def normalize_label(label: str) -> str:
    if not label or "\0" in label or len(label.encode("utf-8")) > MAX_LABEL_BYTES:
        raise DecodeError("channel label is empty, unsafe, or too long")
    normalized = " ".join(label.split()).upper()
    if normalized.startswith("EEG "):
        normalized = normalized[4:].strip()
    normalized = normalized.rstrip(".")
    if not normalized or len(normalized.encode("utf-8")) > MAX_LABEL_BYTES:
        raise DecodeError("channel label is empty after normalization")
    return normalized


def is_known_aux(label: str) -> bool:
    return any(
        label == prefix
        or label.startswith(prefix + " ")
        or label.startswith(prefix + "-")
        for prefix in AUX_PREFIXES
    )


def is_electrode(token: str) -> bool:
    for family in ELECTRODE_FAMILIES:
        if not token.startswith(family):
            continue
        suffix = token[len(family) :]
        if suffix == "Z":
            return True
        if (
            1 <= len(suffix) <= 2
            and all("0" <= character <= "9" for character in suffix)
            and 0 < int(suffix) <= 255
        ):
            return True
    return False


def parse_channel(identity: Identity, presented_index: int) -> ParsedChannel:
    normalized = normalize_label(identity.label)
    if is_known_aux(normalized):
        return ParsedChannel(
            presented_index, identity.stable_id, normalized, 1, 3, "", ""
        )
    pieces = normalized.split("-")
    if len(pieces) == 1 and is_electrode(pieces[0]):
        return ParsedChannel(
            presented_index,
            identity.stable_id,
            normalized,
            0,
            0,
            "E:" + pieces[0],
            "R:MONO",
        )
    if len(pieces) == 2 and is_electrode(pieces[0]):
        positive, negative = pieces
        if negative in REFERENCE_TOKENS:
            return ParsedChannel(
                presented_index,
                identity.stable_id,
                normalized,
                0,
                1,
                "E:" + positive,
                "R:" + negative,
            )
        if is_electrode(negative):
            if positive == negative:
                raise DecodeError("bipolar endpoints must be distinct")
            return ParsedChannel(
                presented_index,
                identity.stable_id,
                normalized,
                0,
                2,
                "E:" + positive,
                "E:" + negative,
            )
    return ParsedChannel(
        presented_index, identity.stable_id, normalized, 1, 3, "", ""
    )


def derive_topology(identities: list[Identity]) -> list[list[tuple[int, int]]]:
    parsed = sorted(
        (parse_channel(identity, index) for index, identity in enumerate(identities)),
        key=ParsedChannel.sort_key,
    )
    if [channel.presented_index for channel in parsed] != list(range(len(identities))):
        raise DecodeError("identity section is not in exact canonical incidence order")

    endpoint_nodes: dict[str, int] = {}
    forest: list[list[tuple[int, int, int]]] = []
    topology: list[list[tuple[int, int]]] = []

    def endpoint_node(endpoint: str) -> int:
        if endpoint in endpoint_nodes:
            return endpoint_nodes[endpoint]
        node = len(forest)
        endpoint_nodes[endpoint] = node
        forest.append([])
        return node

    for canonical_index, channel in enumerate(parsed):
        supports: list[tuple[int, int]] = []
        if channel.partition == 0:
            positive = endpoint_node(channel.positive_endpoint)
            negative = endpoint_node(channel.negative_endpoint)
            path = signed_path(forest, negative, positive)
            if path is None:
                forest[negative].append((positive, canonical_index, 1))
                forest[positive].append((negative, canonical_index, -1))
            elif len(path) <= MAX_SUPPORTS:
                supports = path
        topology.append(supports)
    return topology


def signed_path(
    forest: list[list[tuple[int, int, int]]], start: int, target: int
) -> list[tuple[int, int]] | None:
    if start == target:
        return []
    previous: list[tuple[int, int, int] | None] = [None] * len(forest)
    previous[start] = (start, -1, 0)
    queue = [start]
    head = 0
    found = False
    while head < len(queue) and not found:
        node = queue[head]
        head += 1
        for next_node, channel, coefficient in forest[node]:
            if previous[next_node] is not None:
                continue
            previous[next_node] = (node, channel, coefficient)
            if next_node == target:
                found = True
                break
            queue.append(next_node)
    if previous[target] is None:
        return None
    path: list[tuple[int, int]] = []
    node = target
    while node != start:
        step = previous[node]
        if step is None:
            raise DecodeError("incidence path reconstruction failed")
        parent, channel, coefficient = step
        path.append((channel, coefficient))
        node = parent
    path.reverse()
    return path


class FixedRlsExpert:
    """Independent fixed-integer width-4 RLS implementation."""

    def __init__(self, width: int, bit_depth: int):
        self.width = width
        self.scale = 1 << max(0, bit_depth - 11)
        self.weights_q20 = [0] * width
        self.covariance_q36 = self.identity_covariance()
        self.score_q20 = 0
        self.reset_count = 0

    def identity_covariance(self) -> list[list[int]]:
        return [
            [COVARIANCE_ONE if row == column else 0 for column in range(self.width)]
            for row in range(self.width)
        ]

    def dot_q20(self, features: list[int]) -> int:
        if len(features) != self.width:
            raise DecodeError("fixed RLS feature width mismatch")
        total = 0
        for weight, feature in zip(self.weights_q20, features):
            product = checked_i128(weight * feature, "fixed RLS prediction product")
            total = checked_i128(total + product, "fixed RLS prediction sum")
        return total

    def prediction(self, features: list[int]) -> int:
        return checked_i64(
            round_ratio_away(self.dot_q20(features), WEIGHT_ONE),
            "fixed RLS prediction",
        )

    def observe(self, features: list[int], sample: int) -> None:
        dot_q20 = self.dot_q20(features)
        error_q20 = checked_i128(
            checked_i128(sample * WEIGHT_ONE, "fixed RLS coefficient sample") - dot_q20,
            "fixed RLS coefficient error",
        )

        z_q36: list[int] = []
        for covariance_row in self.covariance_q36:
            value = 0
            for covariance, feature in zip(covariance_row, features):
                product = checked_i128(
                    covariance * feature, "fixed RLS covariance product"
                )
                value = checked_i128(value + product, "fixed RLS covariance sum")
            z_q36.append(value)

        scale_squared = checked_i128(self.scale * self.scale, "fixed RLS scale square")
        regularizer = checked_i128(
            FORGETTING_NUMERATOR * COVARIANCE_ONE,
            "fixed RLS regularizer covariance",
        )
        regularizer = (
            checked_i128(
                regularizer * scale_squared, "fixed RLS regularizer scale"
            )
            // FORGETTING_DENOMINATOR
        )
        denominator = regularizer
        for feature, value in zip(features, z_q36):
            product = checked_i128(feature * value, "fixed RLS denominator product")
            denominator = checked_i128(
                denominator + product, "fixed RLS denominator sum"
            )

        gain_q40: list[int] = []
        reset = denominator <= 0
        if not reset:
            for value in z_q36:
                numerator = checked_i128(value * GAIN_ONE, "fixed RLS gain numerator")
                gain = round_ratio_away(numerator, denominator)
                gain_q40.append(gain)
                if abs(gain) > GAIN_LIMIT:
                    reset = True

        next_weights = list(self.weights_q20)
        next_covariance = [list(row) for row in self.covariance_q36]
        if not reset:
            candidate_weights: list[int] = []
            for weight, gain in zip(self.weights_q20, gain_q40):
                numerator = checked_i128(
                    gain * error_q20, "fixed RLS weight update product"
                )
                adjustment = round_ratio_away(numerator, GAIN_ONE)
                updated = checked_i128(weight + adjustment, "fixed RLS weight update")
                candidate_weights.append(max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, updated)))

            candidate_covariance = [[0] * self.width for _ in range(self.width)]
            for row in range(self.width):
                for column in range(row, self.width):
                    left = checked_i128(
                        gain_q40[row] * z_q36[column],
                        "fixed RLS covariance left product",
                    )
                    right = checked_i128(
                        gain_q40[column] * z_q36[row],
                        "fixed RLS covariance right product",
                    )
                    symmetric_sum = checked_i128(
                        left + right, "fixed RLS covariance symmetric sum"
                    )
                    symmetric = round_ratio_away(symmetric_sum, 2 * GAIN_ONE)
                    difference = checked_i128(
                        self.covariance_q36[row][column] - symmetric,
                        "fixed RLS covariance update difference",
                    )
                    numerator = checked_i128(
                        difference * FORGETTING_DENOMINATOR,
                        "fixed RLS covariance update numerator",
                    )
                    updated = round_ratio_away(numerator, FORGETTING_NUMERATOR)
                    candidate_covariance[row][column] = updated
                    candidate_covariance[column][row] = updated
            reset = any(
                candidate_covariance[index][index] <= 0
                for index in range(self.width)
            ) or any(
                abs(value) > COVARIANCE_LIMIT
                for row in candidate_covariance
                for value in row
            )
            if not reset:
                next_weights = candidate_weights
                next_covariance = candidate_covariance

        if reset:
            next_weights = [0] * self.width
            next_covariance = self.identity_covariance()
            if self.reset_count == U64_MAX:
                raise DecodeError("fixed RLS reset count overflowed")
            self.reset_count += 1

        discounted_numerator = checked_i128(
            FORGETTING_NUMERATOR * self.score_q20, "fixed RLS score discount"
        )
        discounted = round_ratio_away(discounted_numerator, FORGETTING_DENOMINATOR)
        self.score_q20 = checked_i128(
            discounted + abs(error_q20), "fixed RLS score"
        )
        self.weights_q20 = next_weights
        self.covariance_q36 = next_covariance


@dataclass
class ChannelState:
    history: list[int]
    temporal: FixedRlsExpert
    expert_scores: list[int]


class Dix1Session:
    def __init__(
        self,
        topology: list[list[tuple[int, int]]],
        bit_depth: int,
        sample_rate_mhz: int,
        incidence_enabled: bool,
    ):
        self.topology = topology
        self.sample_lags = [
            max(1, (sample_rate_mhz * milliseconds + 500_000) // 1_000_000)
            for milliseconds in TEMPORAL_LAG_MS
        ]
        history_length = max(self.sample_lags)
        if history_length > 256:
            raise DecodeError("temporal history exceeds construction bound")
        magnitude = 1 << (bit_depth - 1)
        self.sample_min = -magnitude
        self.sample_max = magnitude - 1
        self.incidence_enabled = incidence_enabled
        self.states = [
            ChannelState(
                [0] * history_length,
                FixedRlsExpert(4, bit_depth),
                (
                    [INCIDENCE_PRIOR_COST, INCIDENCE_PRIOR_COST, 0, 0]
                    if supports
                    else [0, 0, 0, 0]
                ),
            )
            for supports in topology
        ]
        self.rows = 0

    def prediction_ticket(
        self, channel: int, current: list[int], innovations: list[int]
    ) -> tuple[list[int], list[int], list[bool], int]:
        if len(current) != channel or len(innovations) != channel:
            raise DecodeError("DIX1 prediction prefix is not causal")
        state = self.states[channel]
        features = [state.history[lag - 1] for lag in self.sample_lags]
        temporal = self.clip(state.temporal.prediction(features))
        delta = state.history[0]
        supports = self.topology[channel]
        use_incidence = self.incidence_enabled and bool(supports)
        raw = self.clip(support_sum(supports, current)) if use_incidence else 0
        if use_incidence:
            innovation = self.clip(temporal + support_sum(supports, innovations))
        else:
            innovation = 0
        predictions = [delta, temporal, raw, innovation]
        active = [True, True, True, True] if use_incidence else [True, True, False, False]
        blended = blend_predictions(
            predictions,
            state.expert_scores,
            active,
            self.sample_min,
            self.sample_max,
        )
        return features, predictions, active, blended

    def inverse_row(self, residuals: list[int]) -> list[int]:
        if len(residuals) != len(self.states):
            raise DecodeError("residual row has the wrong channel count")
        current: list[int] = []
        innovations: list[int] = []
        for channel, residual in enumerate(residuals):
            features, predictions, active, blended = self.prediction_ticket(
                channel, current, innovations
            )
            sample = checked_i64(blended + residual, "DIX1 inverse sample")
            self.observe(channel, sample, features, predictions, active)
            current.append(sample)
            innovations.append(
                checked_i64(sample - predictions[1], "DIX1 temporal innovation")
            )
        self.finish_row(current)
        return current

    def forward_row(self, current: list[int]) -> list[int]:
        if len(current) != len(self.states):
            raise DecodeError("sample row has the wrong channel count")
        prefix: list[int] = []
        innovations: list[int] = []
        residuals: list[int] = []
        for channel, sample in enumerate(current):
            features, predictions, active, blended = self.prediction_ticket(
                channel, prefix, innovations
            )
            residuals.append(checked_i64(sample - blended, "DIX1 forward residual"))
            self.observe(channel, sample, features, predictions, active)
            prefix.append(sample)
            innovations.append(
                checked_i64(sample - predictions[1], "DIX1 temporal innovation")
            )
        self.finish_row(current)
        return residuals

    def advance_row(self, current: list[int]) -> None:
        if len(current) != len(self.states):
            raise DecodeError("escape row has the wrong channel count")
        prefix: list[int] = []
        innovations: list[int] = []
        for channel, sample in enumerate(current):
            features, predictions, active, _ = self.prediction_ticket(
                channel, prefix, innovations
            )
            self.observe(channel, sample, features, predictions, active)
            prefix.append(sample)
            innovations.append(
                checked_i64(sample - predictions[1], "DIX1 temporal innovation")
            )
        self.finish_row(current)

    def observe(
        self,
        channel: int,
        sample: int,
        features: list[int],
        predictions: list[int],
        active: list[bool],
    ) -> None:
        if not self.sample_min <= sample <= self.sample_max:
            raise DecodeError("sample exceeds declared bit depth")
        state = self.states[channel]
        state.temporal.observe(features, sample)
        for expert, is_active in enumerate(active):
            if not is_active:
                continue
            residual = checked_i64(
                sample - predictions[expert], "DIX1 expert residual"
            )
            score = state.expert_scores[expert]
            score = score - (score >> SCORE_DECAY_SHIFT) + signed_code_bits(residual) * SCORE_Q
            if not 0 <= score <= U64_MAX:
                raise DecodeError("DIX1 expert score overflowed")
            state.expert_scores[expert] = score

    def finish_row(self, current: list[int]) -> None:
        for state, sample in zip(self.states, current):
            state.history[:] = [sample] + state.history[:-1]
        if self.rows == U64_MAX:
            raise DecodeError("DIX1 row counter overflowed")
        self.rows += 1

    def clip(self, value: int) -> int:
        return max(self.sample_min, min(self.sample_max, value))


def support_sum(supports: list[tuple[int, int]], values: list[int]) -> int:
    total = 0
    for prior, coefficient in supports:
        if prior >= len(values):
            raise DecodeError("incidence support references unavailable channel")
        total = checked_i64(total + coefficient * values[prior], "incidence sum")
    return total


def blend_predictions(
    predictions: list[int],
    scores: list[int],
    active: list[bool],
    sample_min: int,
    sample_max: int,
) -> int:
    numerator = 0
    denominator = 0
    for prediction, score, is_active in zip(predictions, scores, active):
        if not is_active:
            continue
        weight = max(1, MIX_WEIGHT_NUMERATOR // (score // SCORE_Q + 1))
        numerator += prediction * weight
        denominator += weight
    prediction = round_ratio_away(numerator, denominator)
    return max(sample_min, min(sample_max, prediction))


def signed_code_bits(value: int) -> int:
    magnitude = abs(value)
    if value >= 0:
        zigzag = min(U64_MAX, magnitude * 2)
    else:
        zigzag = min(U64_MAX, magnitude * 2) - 1
    return zigzag.bit_length() + 1


@dataclass
class Counts:
    zeros: int = 0
    ones: int = 0

    def probability_one(self) -> int:
        numerator = (2 * self.ones + 1) * CDF_TOTAL
        denominator = 2 * (self.zeros + self.ones + 1)
        return max(1, min(CDF_TOTAL - 1, numerator // denominator))

    def observe(self, bit: int) -> None:
        if bit == 0:
            self.zeros += 1
        elif bit == 1:
            self.ones += 1
        else:
            raise DecodeError("entropy bit is not binary")
        if self.zeros > 0xFFFFFFFF or self.ones > 0xFFFFFFFF:
            raise DecodeError("entropy context count overflowed")


class ChannelModel:
    def __init__(self):
        self.nonzero = Counts()
        self.sign = Counts()
        self.exponent = [Counts() for _ in range(32)]
        self.mantissa = [Counts() for _ in range(32)]


class BinaryRansEncoder:
    def __init__(self, max_events: int):
        if max_events <= 0:
            raise DecodeError("binary rANS event bound must be nonzero")
        self.events = array("H")
        self.max_events = max_events

    def push(self, bit: int, probability_one: int) -> None:
        if bit not in (0, 1) or not 1 <= probability_one < CDF_TOTAL:
            raise DecodeError("binary rANS event is invalid")
        if len(self.events) >= self.max_events:
            raise DecodeError("binary rANS event bound exceeded")
        self.events.append((probability_one << 1) | bit)

    def finish(self) -> bytes:
        state = RANS_L
        renormalized = bytearray()
        for packed_event in reversed(self.events):
            bit = packed_event & 1
            probability_one = packed_event >> 1
            frequency_zero = CDF_TOTAL - probability_one
            if bit == 1:
                start, frequency = frequency_zero, probability_one
            else:
                start, frequency = 0, frequency_zero
            maximum = ((RANS_L >> CDF_BITS) << 8) * frequency
            while state >= maximum:
                renormalized.append(state & 0xFF)
                state >>= 8
            state = ((state // frequency) << CDF_BITS) + (state % frequency) + start
        if not RANS_L <= state < RANS_L << 8:
            raise DecodeError("binary rANS final state exceeds u32 bound")
        return state.to_bytes(4, "little") + bytes(reversed(renormalized))


class BinaryRansDecoder:
    def __init__(self, payload: bytes, max_events: int):
        if len(payload) < 4 or max_events <= 0:
            raise DecodeError("binary rANS stream is truncated or unbounded")
        self.payload = payload
        self.state = u32le(payload, 0)
        if not RANS_L <= self.state < RANS_L << 8:
            raise DecodeError("binary rANS initial state is invalid")
        self.offset = 4
        self.event_count = 0
        self.max_events = max_events

    def read(self, probability_one: int) -> int:
        if not 1 <= probability_one < CDF_TOTAL:
            raise DecodeError("binary rANS probability is invalid")
        if self.event_count >= self.max_events:
            raise DecodeError("binary rANS event bound exceeded")
        frequency_zero = CDF_TOTAL - probability_one
        cumulative = self.state & (CDF_TOTAL - 1)
        if cumulative < frequency_zero:
            bit, start, frequency = 0, 0, frequency_zero
        else:
            bit, start, frequency = 1, frequency_zero, probability_one
        self.state = frequency * (self.state >> CDF_BITS) + cumulative - start
        while self.state < RANS_L:
            if self.offset >= len(self.payload):
                raise DecodeError("binary rANS renormalization is truncated")
            self.state = (self.state << 8) | self.payload[self.offset]
            self.offset += 1
        self.event_count += 1
        return bit

    def finish(self) -> None:
        if self.offset != len(self.payload) or self.state != RANS_L:
            raise DecodeError("binary rANS stream is noncanonical")


class EntropyDecoder:
    def __init__(self, payload: bytes, channels: int, values: int, bit_depth: int):
        self.bit_depth = bit_depth
        self.coder = BinaryRansDecoder(payload, values * (2 * bit_depth + 1))
        self.models = [ChannelModel() for _ in range(channels)]

    def read_context(self, context: Counts) -> int:
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
                raise DecodeError("residual exponent exceeds bit depth")
        sign = self.read_context(model.sign)
        magnitude = 1 << exponent
        for position in reversed(range(exponent)):
            magnitude |= self.read_context(model.mantissa[position]) << position
        return magnitude if sign == 0 else -magnitude


class EntropyEncoder:
    def __init__(self, channels: int, values: int, bit_depth: int):
        self.bit_depth = bit_depth
        self.coder = BinaryRansEncoder(values * (2 * bit_depth + 1))
        self.models = [ChannelModel() for _ in range(channels)]

    def push_context(self, context: Counts, bit: int) -> None:
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
            raise DecodeError("residual magnitude exceeds bit depth")
        for position in range(exponent + 1):
            self.push_context(model.exponent[position], int(position < exponent))
        self.push_context(model.sign, int(value < 0))
        for position in reversed(range(exponent)):
            self.push_context(model.mantissa[position], (magnitude >> position) & 1)


def read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(payload):
            raise DecodeError("delta varint is truncated")
        byte = payload[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise DecodeError("delta varint overflows u64")
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            if offset - start > 1 and byte == 0:
                raise DecodeError("delta varint is noncanonical")
            return value, offset
    raise DecodeError("delta varint is too long")


def unzigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def encode_raw(signal: list[list[int]]) -> bytes:
    payload = bytearray()
    for channel in signal:
        for sample in channel:
            if not -(1 << 31) <= sample < 1 << 31:
                raise DecodeError("raw sample exceeds i32")
            payload.extend(struct.pack("<i", sample))
    return bytes(payload)


def decode_raw(payload: bytes, channels: int, rows: int) -> list[list[int]]:
    if len(payload) != channels * rows * 4:
        raise DecodeError("raw block length is invalid")
    signal: list[list[int]] = []
    offset = 0
    for _ in range(channels):
        channel: list[int] = []
        for _ in range(rows):
            channel.append(struct.unpack_from("<i", payload, offset)[0])
            offset += 4
        signal.append(channel)
    return signal


def encode_varint(value: int, payload: bytearray) -> None:
    while value >= 0x80:
        payload.append((value & 0x7F) | 0x80)
        value >>= 7
    payload.append(value)


def zigzag(value: int) -> int:
    checked_i64(value, "delta")
    return value * 2 if value >= 0 else -value * 2 - 1


def encode_delta(signal: list[list[int]]) -> bytes:
    payload = bytearray()
    for channel in signal:
        previous = 0
        for index, sample in enumerate(channel):
            delta = sample if index == 0 else checked_i64(sample - previous, "delta")
            encode_varint(zigzag(delta), payload)
            previous = sample
    return bytes(payload)


def decode_delta(payload: bytes, channels: int, rows: int) -> list[list[int]]:
    values = channels * rows
    if not values <= len(payload) <= values * 5:
        raise DecodeError("delta block length is outside bounds")
    offset = 0
    signal: list[list[int]] = []
    for _ in range(channels):
        channel: list[int] = []
        previous = 0
        for index in range(rows):
            encoded, offset = read_varint(payload, offset)
            delta = unzigzag(encoded)
            sample = delta if index == 0 else checked_i64(previous + delta, "delta sample")
            if not -(1 << 31) <= sample < 1 << 31:
                raise DecodeError("delta sample exceeds i32")
            channel.append(sample)
            previous = sample
        signal.append(channel)
    if offset != len(payload):
        raise DecodeError("delta block has trailing bytes")
    return signal


def canonical_block_payload(
    profile: int,
    stable_signal: list[list[int]],
    identities: list[Identity],
    block_start_session: Dix1Session,
    bit_depth: int,
) -> tuple[int, bytes]:
    raw = encode_raw(stable_signal)
    delta = encode_delta(stable_signal)
    if profile in (0, 4, 5):
        entropy_session = copy.deepcopy(block_start_session)
        rows = len(stable_signal[0])
        entropy = EntropyEncoder(len(stable_signal), len(stable_signal) * rows, bit_depth)
        for row in range(rows):
            canonical_row = [
                stable_signal[identity.stable_id][row] for identity in identities
            ]
            for channel, residual in enumerate(entropy_session.forward_row(canonical_row)):
                entropy.push_value(channel, residual)
        entropy_payload = entropy.coder.finish()
    else:
        entropy_payload = b""

    candidates = {
        0: ((0, raw), (1, delta), (2, entropy_payload)),
        1: ((0, raw), (1, delta)),
        2: ((0, raw),),
        3: ((1, delta),),
        4: ((2, entropy_payload),),
        5: ((3, entropy_payload),),
    }[profile]
    return min(candidates, key=lambda candidate: (len(candidate[1]), candidate[0]))


def permitted(profile: int, mode: int) -> bool:
    return {
        0: mode in (0, 1, 2),
        1: mode in (0, 1),
        2: mode == 0,
        3: mode == 1,
        4: mode == 2,
        5: mode == 3,
    }.get(profile, False)


def decode_packet(packet: bytes) -> dict[str, object]:
    if not HEADER_LEN <= len(packet) <= MAX_PACKET_BYTES:
        raise DecodeError("packet length is outside bounds")
    if packet[:7] != b"LMO1\x03\x00\x03" or packet[7:11] != b"DIX1":
        raise DecodeError("DIX1 envelope or magic is invalid")
    if packet[11] != 2 or packet[12] != 1:
        raise DecodeError("DIX1 body version or flags are invalid")

    bit_depth = packet[13]
    profile = packet[14]
    channels = u16le(packet, 15)
    tile_count = u16le(packet, 17)
    samples = u32le(packet, 19)
    sample_rate_mhz = u32le(packet, 23)
    model_id = u32le(packet, 27)
    identity_len = u32le(packet, 63)
    topology_len = u32le(packet, 67)
    directory_len = u32le(packet, 71)
    payload_len = u32le(packet, 75)
    decoded_crc = u32le(packet, 79)
    stored_packet_crc = u32le(packet, 83)

    values = channels * samples
    if not (
        1 <= channels <= MAX_CHANNELS
        and 1 <= samples <= MAX_SAMPLES
        and values <= MAX_VALUES
        and 1 <= bit_depth <= 32
        and 1 <= sample_rate_mhz <= MAX_SAMPLE_RATE_MHZ
    ):
        raise DecodeError("DIX1 dimensions are outside bounds")
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
        raise DecodeError("DIX1 construction header is invalid")

    identity_start = HEADER_LEN
    identity_end = identity_start + identity_len
    topology_end = identity_end + topology_len
    directory_end = topology_end + directory_len
    packet_end = directory_end + payload_len
    if packet_end != len(packet):
        raise DecodeError("DIX1 section lengths do not cover packet")
    if stored_packet_crc != packet_crc32c(packet):
        raise DecodeError("DIX1 packet CRC32C mismatch")

    identities = parse_identities(packet[identity_start:identity_end], channels)
    topology = parse_topology(packet[identity_end:topology_end], channels)
    if topology != derive_topology(identities):
        raise DecodeError("topology does not match deterministic derivation incidence")
    directory = packet[topology_end:directory_end]
    payload = packet[directory_end:]
    entries: list[tuple[int, int]] = []
    payload_sum = 0
    for entry in range(blocks):
        offset = entry * DIRECTORY_ENTRY_LEN
        mode = directory[offset]
        length = u32le(directory, offset + 1)
        if mode not in range(4) or not permitted(profile, mode):
            raise DecodeError("block mode is not permitted by profile")
        payload_sum += length
        if payload_sum > len(payload):
            raise DecodeError("block payload lengths exceed payload section")
        entries.append((mode, length))
    if payload_sum != len(payload):
        raise DecodeError("directory does not cover payload exactly")

    session = Dix1Session(topology, bit_depth, sample_rate_mhz, profile != 5)
    stable_signal: list[list[int]] = [[] for _ in range(channels)]
    tile_modes: list[int] = []
    payload_offset = 0
    event_count = 0

    for block, (mode, length) in enumerate(entries):
        rows = min(BLOCK_ROWS, samples - block * BLOCK_ROWS)
        end = payload_offset + length
        block_payload = payload[payload_offset:end]
        block_start_session = copy.deepcopy(session)
        if mode == 0:
            stable_block = decode_raw(block_payload, channels, rows)
            advance_escape_block(session, identities, stable_block, rows)
        elif mode == 1:
            stable_block = decode_delta(block_payload, channels, rows)
            advance_escape_block(session, identities, stable_block, rows)
        else:
            entropy = EntropyDecoder(block_payload, channels, channels * rows, bit_depth)
            stable_block = [[] for _ in range(channels)]
            for _ in range(rows):
                residuals = [entropy.read_value(channel) for channel in range(channels)]
                canonical_row = session.inverse_row(residuals)
                for canonical_channel, sample in enumerate(canonical_row):
                    stable_block[identities[canonical_channel].stable_id].append(sample)
            entropy.coder.finish()
            event_count += entropy.coder.event_count
        canonical_mode, canonical_payload = canonical_block_payload(
            profile,
            stable_block,
            identities,
            block_start_session,
            bit_depth,
        )
        if mode != canonical_mode or block_payload != canonical_payload:
            raise DecodeError("block mode or payload is not byte-canonical for profile")
        for stable_id in range(channels):
            stable_signal[stable_id].extend(stable_block[stable_id])
        tile_modes.append(mode)
        payload_offset = end

    if payload_offset != len(payload) or any(len(channel) != samples for channel in stable_signal):
        raise DecodeError("block decode did not cover the window")

    decoded_bytes = bytearray()
    sample_min = -(1 << (bit_depth - 1))
    sample_max = (1 << (bit_depth - 1)) - 1
    for channel in stable_signal:
        for sample in channel:
            if not sample_min <= sample <= sample_max:
                raise DecodeError("decoded sample exceeds declared bit depth")
            if not -(1 << 31) <= sample < 1 << 31:
                raise DecodeError("decoded sample exceeds i32")
            decoded_bytes.extend(struct.pack("<i", sample))
    if crc32c(decoded_bytes) != decoded_crc:
        raise DecodeError("decoded-sample CRC32C mismatch")

    return {
        "sample_rate_mhz": sample_rate_mhz,
        "bit_depth": bit_depth,
        "stable_ids": [identity.stable_id for identity in sorted(identities, key=lambda item: item.stable_id)],
        "labels": [identity.label for identity in sorted(identities, key=lambda item: item.stable_id)],
        "tile_modes": tile_modes,
        "event_count": event_count,
        "samples": stable_signal,
    }


def advance_escape_block(
    session: Dix1Session,
    identities: list[Identity],
    stable_block: list[list[int]],
    rows: int,
) -> None:
    for row in range(rows):
        canonical_row = [stable_block[identity.stable_id][row] for identity in identities]
        session.advance_row(canonical_row)


def main() -> int:
    packet = sys.stdin.buffer.read(MAX_PACKET_BYTES + 1)
    if len(packet) > MAX_PACKET_BYTES:
        raise DecodeError("packet exceeds 64 MiB")
    result = decode_packet(packet)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DecodeError, MemoryError, OverflowError, struct.error) as error:
        print(f"DIX1 reference decode failed: {error}", file=sys.stderr)
        raise SystemExit(2)
