"""LamQuant file formats: .lmq (neural compressed) and .lml (lossless).

Two file types, two extensions, clean API.

.lmq — LamQuant neural compressed. rANS-coded FSQ tokens from the ternary
       encoder. Requires a GPU decoder (Tier 5/6/7) to reconstruct signal.
.lml — LamQuant lossless. Integer-exact lifting + LPC + Golomb-Rice.
       Self-decodable, no neural network needed.

File structure (both types):
    ┌────────────────────┐
    │ File Header (64B)  │  magic, version, session metadata
    ├────────────────────┤
    │ Window 0 packet    │  per-window header + payload + CRC32
    ├────────────────────┤
    │ Window 1 packet    │
    ├────────────────────┤
    │ ...                │
    ├────────────────────┤
    │ Window N packet    │
    ├────────────────────┤
    │ Index table (opt)  │  offsets for random access by timestamp
    ├────────────────────┤
    │ Footer (12B)       │  index_offset (u64) + footer_magic (LQFT)
    └────────────────────┘

Magic bytes:
    .lmq:  b'LQN1'  (0x4C514E31) — LamQuant Neural v1
    .lml:  b'LQL1'  (0x4C514C31) — LamQuant Lossless v1

Usage:
    import lamquant_codec as lq

    # Write neural compressed
    with lq.NeuralWriter("recording.lmq", channels=21, rate=250) as w:
        w.write_window(tokens, snn_levels, timestamp)

    # Write lossless
    with lq.LosslessWriter("recording.lml", channels=21, rate=250) as w:
        w.write_window(signal_int16, timestamp)

    # Read either format
    with lq.open("recording.lmq") as r:
        for window in r:
            print(window.timestamp, window.payload_size)
"""

import struct
import time
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Iterator, BinaryIO


# ============================================================
# Constants
# ============================================================

MAGIC_NEURAL  = b'LQN1'   # 0x4C514E31 — .lmq file magic
MAGIC_LOSSLESS = b'LQL1'  # 0x4C514C31 — .lml file magic
MAGIC_FOOTER  = b'LQFT'   # footer sentinel
LMQ1_MAGIC    = b'LMQ1'   # payload magic — uniform FSQ
LMQ3_MAGIC    = b'LMQ3'   # payload magic — adaptive per-timestep FSQ
FILE_HEADER_SIZE = 64
FOOTER_SIZE = 12           # 8B index_offset (uint64) + 4B footer_magic (LQFT)
FORMAT_VERSION = 1

# Flag bits for .lmq window header
FLAG_HAS_SNN     = 0x0001
FLAG_HAS_LPC     = 0x0002
FLAG_ADAPTIVE_FSQ = 0x0004
FLAG_HAS_DETAILS = 0x0008

# ============================================================
# Data classes
# ============================================================

@dataclass
class FileHeader:
    """64-byte file header for both .lmq and .lml."""
    magic: bytes = MAGIC_NEURAL        # 4B
    version: int = FORMAT_VERSION      # 1B
    channels: int = 21                 # 1B
    sample_rate: int = 250             # 2B
    window_samples: int = 2500         # 2B
    session_id: int = 0                # 4B  (random or user-set)
    start_time_us: int = 0             # 8B  (unix microseconds)
    encoder_version: bytes = b'\x07\x70'  # 2B  (0x0770 = v7.7.0)
    decoder_tier_hint: int = 0         # 1B  (0=any, 5/6/7=specific tier)
    reserved: bytes = b'\x00' * 39     # 39B padding to 64

    def pack(self) -> bytes:
        buf = struct.pack('<4sBBHHIq2sB',
                          self.magic,
                          self.version,
                          self.channels,
                          self.sample_rate,
                          self.window_samples,
                          self.session_id,
                          self.start_time_us,
                          self.encoder_version,
                          self.decoder_tier_hint)
        # buf is 25 bytes, pad to 64
        return buf + b'\x00' * (FILE_HEADER_SIZE - len(buf))

    @classmethod
    def unpack(cls, data: bytes) -> 'FileHeader':
        if len(data) < FILE_HEADER_SIZE:
            raise ValueError(f"Data too short for LML header: {len(data)} < {FILE_HEADER_SIZE} bytes")
        magic, ver, ch, rate, wlen, sid, ts, enc_ver, tier = struct.unpack(
            '<4sBBHHIq2sB', data[:25])
        return cls(
            magic=magic, version=ver, channels=ch, sample_rate=rate,
            window_samples=wlen, session_id=sid, start_time_us=ts,
            encoder_version=enc_ver, decoder_tier_hint=tier,
        )


@dataclass
class NeuralWindowHeader:
    """26-byte per-window header for .lmq packets."""
    version: int = 1                   # 1B
    channels: int = 21                 # 1B
    sample_rate: int = 250             # 2B
    window_samples: int = 2500         # 2B
    timestamp_us: int = 0              # 8B
    fsq_levels: bytes = b'\x02' * 10   # 10B  (per-block L=2/3/5 map)
    flags: int = 0                     # 2B

    STRUCT = '<BBHHq10sH'
    SIZE = 26

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT,
                           self.version, self.channels, self.sample_rate,
                           self.window_samples, self.timestamp_us,
                           self.fsq_levels[:10].ljust(10, b'\x00'),
                           self.flags)

    @classmethod
    def unpack(cls, data: bytes) -> 'NeuralWindowHeader':
        v, ch, rate, wlen, ts, fsq, flags = struct.unpack(cls.STRUCT, data[:cls.SIZE])
        return cls(version=v, channels=ch, sample_rate=rate,
                   window_samples=wlen, timestamp_us=ts,
                   fsq_levels=fsq, flags=flags)


@dataclass
class LosslessWindowHeader:
    """22-byte per-window header for .lml packets."""
    version: int = 1                   # 1B
    channels: int = 21                 # 1B
    sample_rate: int = 250             # 2B
    window_samples: int = 2500         # 2B
    timestamp_us: int = 0              # 8B
    lpc_orders: bytes = b'\x02' * 4    # 4B  (per-subband orders)
    gr_k_values: bytes = b'\x00' * 4   # 4B  (per-subband adaptive k)

    STRUCT = '<BBHHq4s4s'
    SIZE = 22

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT,
                           self.version, self.channels, self.sample_rate,
                           self.window_samples, self.timestamp_us,
                           self.lpc_orders[:4].ljust(4, b'\x00'),
                           self.gr_k_values[:4].ljust(4, b'\x00'))

    @classmethod
    def unpack(cls, data: bytes) -> 'LosslessWindowHeader':
        v, ch, rate, wlen, ts, lpc, gr = struct.unpack(cls.STRUCT, data[:cls.SIZE])
        return cls(version=v, channels=ch, sample_rate=rate,
                   window_samples=wlen, timestamp_us=ts,
                   lpc_orders=lpc, gr_k_values=gr)


@dataclass
class IndexEntry:
    """8-byte index table entry: timestamp + file offset."""
    timestamp_us: int = 0   # 8B relative to session start
    offset: int = 0         # 8B byte offset from file start

    STRUCT = '<qQ'
    SIZE = 16

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT, self.timestamp_us, self.offset)

    @classmethod
    def unpack(cls, data: bytes) -> 'IndexEntry':
        ts, off = struct.unpack(cls.STRUCT, data[:cls.SIZE])
        return cls(timestamp_us=ts, offset=off)


@dataclass
class Window:
    """A decoded window from a .lmq or .lml file."""
    timestamp_us: int
    header: object           # NeuralWindowHeader or LosslessWindowHeader
    payload: bytes           # raw compressed payload (between header and CRC)
    payload_size: int
    file_offset: int         # byte offset in the file
    mode: str                # 'neural' or 'lossless'

    @property
    def timestamp(self) -> float:
        """Timestamp in seconds (float)."""
        return self.timestamp_us / 1_000_000.0

    def decode(self) -> np.ndarray:
        """Decode lossless window to signal. Only works for .lml windows."""
        if self.mode != 'lossless':
            raise RuntimeError(
                "Neural windows require an external decoder. "
                "Use lq.Decoder(tier=N).decode(window) instead."
            )
        # Import here to avoid circular deps
        from lamquant_codec.codec import LosslessCodec
        codec = LosslessCodec(klt_matrix=None, n_levels=3)
        return codec.decompress(self.payload).astype(np.float64)


# ============================================================
# Writers
# ============================================================

class NeuralWriter:
    """Write .lmq files (neural compressed EEG).

    Usage:
        with NeuralWriter("recording.lmq", channels=21, rate=250) as w:
            w.write_window(compressed_payload, snn_levels=levels, timestamp_us=ts)
    """

    def __init__(self, path: str, channels: int = 21, rate: int = 250,
                 window_samples: int = 2500, session_id: int = 0,
                 decoder_tier_hint: int = 0):
        self.path = path
        self.channels = channels
        self.rate = rate
        self.window_samples = window_samples
        self.session_id = session_id or (zlib.crc32(str(time.time()).encode()) & 0xFFFFFFFF)
        self.decoder_tier_hint = decoder_tier_hint
        self._f: Optional[BinaryIO] = None
        self._index: list[IndexEntry] = []
        self._window_count = 0

    def __enter__(self):
        self._f = open(self.path, 'wb')
        hdr = FileHeader(
            magic=MAGIC_NEURAL,
            channels=self.channels,
            sample_rate=self.rate,
            window_samples=self.window_samples,
            session_id=self.session_id,
            start_time_us=int(time.time() * 1_000_000),
            decoder_tier_hint=self.decoder_tier_hint,
        )
        self._f.write(hdr.pack())
        return self

    def write_window(self, payload: bytes, *,
                     snn_levels: Optional[bytes] = None,
                     fsq_levels: Optional[bytes] = None,
                     timestamp_us: Optional[int] = None,
                     flags: int = 0):
        """Write one neural compressed window.

        Args:
            payload: raw compressed bytes (rANS tokens + freq tables)
            snn_levels: 10-byte SNN level map (optional)
            fsq_levels: 10-byte FSQ level map (optional, overrides snn_levels)
            timestamp_us: unix microseconds (default: now)
            flags: bitfield (FLAG_HAS_SNN, FLAG_HAS_LPC, etc.)
        """
        if self._f is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1_000_000)

        # Auto-detect LMQ3 (adaptive FSQ) payloads. The per-timestep
        # level schedule is canonical in the payload — the 10-byte
        # header field MUST be zero. Per Q1=A in the v7.7.1 plan.
        # NOTE: any caller-supplied `fsq_levels` / `snn_levels` is
        # ignored for LMQ3 payloads (override is intentional; the
        # payload is the single source of truth).
        is_adaptive = len(payload) >= 4 and bytes(payload[:4]) == LMQ3_MAGIC
        if is_adaptive:
            flags |= FLAG_ADAPTIVE_FSQ
            fsq = b'\x00' * 10
        else:
            fsq = fsq_levels or snn_levels or (b'\x02' * 10)
        if snn_levels is not None:
            flags |= FLAG_HAS_SNN

        whdr = NeuralWindowHeader(
            channels=self.channels,
            sample_rate=self.rate,
            window_samples=self.window_samples,
            timestamp_us=ts,
            fsq_levels=fsq[:10],
            flags=flags,
        )

        offset = self._f.tell()
        self._index.append(IndexEntry(timestamp_us=ts, offset=offset))

        # Write: [payload_size:4B][header:26B][payload:NB][crc32:4B]
        hdr_bytes = whdr.pack()
        payload_len = struct.pack('<I', len(payload))
        crc = struct.pack('<I', zlib.crc32(hdr_bytes + payload) & 0xFFFFFFFF)

        self._f.write(payload_len)
        self._f.write(hdr_bytes)
        self._f.write(payload)
        self._f.write(crc)
        self._window_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._f is None:
            return
        # Write index table
        index_offset = self._f.tell()
        n_entries = len(self._index)
        self._f.write(struct.pack('<I', n_entries))
        for entry in self._index:
            self._f.write(entry.pack())
        # Footer: index_offset + magic
        self._f.write(struct.pack('<Q', index_offset))
        self._f.write(MAGIC_FOOTER)
        self._f.close()
        self._f = None


class LosslessWriter:
    """Write .lml files (lossless compressed EEG).

    Usage:
        with LosslessWriter("recording.lml", channels=21, rate=250) as w:
            w.write_window(signal_int16, timestamp_us=ts)
    """

    def __init__(self, path: str, channels: int = 21, rate: int = 250,
                 window_samples: int = 2500, session_id: int = 0):
        self.path = path
        self.channels = channels
        self.rate = rate
        self.window_samples = window_samples
        self.session_id = session_id or (zlib.crc32(str(time.time()).encode()) & 0xFFFFFFFF)
        self._f: Optional[BinaryIO] = None
        self._index: list[IndexEntry] = []
        self._window_count = 0

    def __enter__(self):
        self._f = open(self.path, 'wb')
        hdr = FileHeader(
            magic=MAGIC_LOSSLESS,
            channels=self.channels,
            sample_rate=self.rate,
            window_samples=self.window_samples,
            session_id=self.session_id,
            start_time_us=int(time.time() * 1_000_000),
            decoder_tier_hint=0,  # lossless needs no decoder
        )
        self._f.write(hdr.pack())
        return self

    def write_window(self, signal: np.ndarray, *,
                     timestamp_us: Optional[int] = None,
                     lpc_orders: Optional[bytes] = None,
                     gr_k_values: Optional[bytes] = None):
        """Write one lossless window.

        Args:
            signal: [channels, samples] integer or float EEG signal.
                    Rounded to int64 internally for lossless encoding.
            timestamp_us: unix microseconds (default: now)
            lpc_orders: 4-byte per-subband LPC orders (default: all 2)
            gr_k_values: 4-byte per-subband GR k params (auto-computed if None)
        """
        if self._f is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1_000_000)

        # Compress the signal using LosslessCodec
        from lamquant_codec.codec import LosslessCodec
        codec = LosslessCodec(klt_matrix=None, n_levels=3)
        payload = codec.compress(signal.astype(np.float64))

        whdr = LosslessWindowHeader(
            channels=self.channels,
            sample_rate=self.rate,
            window_samples=self.window_samples,
            timestamp_us=ts,
            lpc_orders=lpc_orders or b'\x02\x02\x02\x02',
            gr_k_values=gr_k_values or b'\x00\x00\x00\x00',
        )

        offset = self._f.tell()
        self._index.append(IndexEntry(timestamp_us=ts, offset=offset))

        # Write: [payload_size:4B][header:22B][payload:NB][crc32:4B]
        hdr_bytes = whdr.pack()
        payload_len = struct.pack('<I', len(payload))
        crc = struct.pack('<I', zlib.crc32(hdr_bytes + payload) & 0xFFFFFFFF)

        self._f.write(payload_len)
        self._f.write(hdr_bytes)
        self._f.write(payload)
        self._f.write(crc)
        self._window_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._f is None:
            return
        # Write index table
        index_offset = self._f.tell()
        n_entries = len(self._index)
        self._f.write(struct.pack('<I', n_entries))
        for entry in self._index:
            self._f.write(entry.pack())
        # Footer
        self._f.write(struct.pack('<Q', index_offset))
        self._f.write(MAGIC_FOOTER)
        self._f.close()
        self._f = None


# ============================================================
# Reader
# ============================================================

class LMQReader:
    """Read .lmq or .lml files. Iterable, supports random access.

    Usage:
        with lq.open("recording.lmq") as r:
            print(r.file_header)
            for window in r:
                print(window.timestamp, window.payload_size)

        # Random access
        with lq.open("recording.lml") as r:
            window = r.seek_timestamp(5.0)  # jump to ~5 seconds
            signal = window.decode()
    """

    def __init__(self, path: str):
        self.path = path
        self._f: Optional[BinaryIO] = None
        self.file_header: Optional[FileHeader] = None
        self.mode: str = ''
        self._index: list[IndexEntry] = []
        self._data_start: int = FILE_HEADER_SIZE
        self._index_loaded = False

    def __enter__(self):
        self._f = open(self.path, 'rb')
        hdr_data = self._f.read(FILE_HEADER_SIZE)
        if len(hdr_data) < FILE_HEADER_SIZE:
            raise ValueError(f"File too small: {len(hdr_data)} bytes")
        self.file_header = FileHeader.unpack(hdr_data)

        if self.file_header.magic == MAGIC_NEURAL:
            self.mode = 'neural'
        elif self.file_header.magic == MAGIC_LOSSLESS:
            self.mode = 'lossless'
        else:
            raise ValueError(
                f"Unknown magic: {self.file_header.magic!r}. "
                f"Expected {MAGIC_NEURAL!r} (.lmq) or {MAGIC_LOSSLESS!r} (.lml)")

        self._data_start = FILE_HEADER_SIZE
        self._data_end = None  # set by _load_index
        self._load_index()
        return self

    def _load_index(self):
        """Load the index table from the footer (if present).

        Footer layout: [8B index_offset (uint64 LE)] [4B LQFT magic] = 12 bytes.
        AUDIT (2026-04-28): Simplified from two-pass seek to single read now
        that FOOTER_SIZE is correct (was 8, now 12).
        """
        if self._f is None:
            return
        pos = self._f.tell()
        try:
            self._f.seek(-FOOTER_SIZE, 2)
            tail = self._f.read(FOOTER_SIZE)
            if len(tail) < FOOTER_SIZE:
                return
            if tail[8:12] != MAGIC_FOOTER:
                return
            index_offset_full = struct.unpack('<Q', tail[:8])[0]

            self._f.seek(index_offset_full)
            n_entries = struct.unpack('<I', self._f.read(4))[0]
            self._index = []
            for _ in range(n_entries):
                entry_data = self._f.read(IndexEntry.SIZE)
                if len(entry_data) < IndexEntry.SIZE:
                    break
                self._index.append(IndexEntry.unpack(entry_data))
            self._index_loaded = True
            self._data_end = index_offset_full
        except Exception as exc:
            import warnings
            warnings.warn(f"LMQ index corrupt, falling back to sequential: {exc}")
            self._index = []
        finally:
            self._f.seek(pos)

    @property
    def window_count(self) -> int:
        """Number of windows (from index if available)."""
        return len(self._index) if self._index_loaded else -1

    def __iter__(self) -> Iterator[Window]:
        if self._f is None:
            raise RuntimeError("Reader not opened. Use 'with' statement.")
        self._f.seek(self._data_start)
        return self

    def __next__(self) -> Window:
        return self._read_window()

    def _read_window(self) -> Window:
        """Read one window at the current file position."""
        if self._f is None:
            raise StopIteration

        file_offset = self._f.tell()

        # Stop before the index table
        if self._data_end is not None and file_offset >= self._data_end:
            raise StopIteration

        # Read payload size (4 bytes)
        size_data = self._f.read(4)
        if len(size_data) < 4:
            raise StopIteration
        payload_size = struct.unpack('<I', size_data)[0]

        if self.mode == 'neural':
            hdr_size = NeuralWindowHeader.SIZE
            hdr_data = self._f.read(hdr_size)
            if len(hdr_data) < hdr_size:
                raise StopIteration
            header = NeuralWindowHeader.unpack(hdr_data)
            ts = header.timestamp_us
        else:
            hdr_size = LosslessWindowHeader.SIZE
            hdr_data = self._f.read(hdr_size)
            if len(hdr_data) < hdr_size:
                raise StopIteration
            header = LosslessWindowHeader.unpack(hdr_data)
            ts = header.timestamp_us

        # Read payload
        payload = self._f.read(payload_size)
        if len(payload) < payload_size:
            raise StopIteration

        # Read and verify CRC32
        crc_data = self._f.read(4)
        if len(crc_data) < 4:
            raise StopIteration
        stored_crc = struct.unpack('<I', crc_data)[0]
        computed_crc = zlib.crc32(hdr_data + payload) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC mismatch at offset {file_offset}: "
                f"stored=0x{stored_crc:08X}, computed=0x{computed_crc:08X}")

        return Window(
            timestamp_us=ts,
            header=header,
            payload=payload,
            payload_size=payload_size,
            file_offset=file_offset,
            mode=self.mode,
        )

    def seek_window(self, index: int) -> Window:
        """Random access: jump to window by index."""
        if not self._index_loaded or index >= len(self._index):
            raise IndexError(f"Window index {index} out of range (have {len(self._index)})")
        entry = self._index[index]
        self._f.seek(entry.offset)
        return self._read_window()

    def seek_timestamp(self, seconds: float) -> Window:
        """Random access: jump to the window closest to the given timestamp."""
        if not self._index_loaded or not self._index:
            raise RuntimeError("No index table — cannot seek by timestamp")
        target_us = int(seconds * 1_000_000) + self.file_header.start_time_us
        # Binary search
        lo, hi = 0, len(self._index) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._index[mid].timestamp_us < target_us:
                lo = mid + 1
            else:
                hi = mid
        return self.seek_window(lo)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._f is not None:
            self._f.close()
            self._f = None


def open_file(path: str) -> LMQReader:
    """Open a .lmq or .lml file for reading. Auto-detects format from magic.

    Usage:
        with lq.open("recording.lmq") as r:
            for window in r:
                ...
    """
    return LMQReader(path)


def convert(src: str, dst: str, decoder=None, encoder=None):
    """Convert between .lmq and .lml formats.

    .lmq -> .lml: requires a decoder to reconstruct the signal
    .lml -> .lmq: requires an encoder to compress the signal

    Args:
        src: source file path
        dst: destination file path
        decoder: callable(payload) -> np.ndarray for .lmq->.lml
        encoder: callable(signal) -> bytes for .lml->.lmq
    """
    with open_file(src) as reader:
        src_mode = reader.mode
        hdr = reader.file_header

        if src_mode == 'neural' and dst.endswith('.lml'):
            if decoder is None:
                raise ValueError("decoder required for .lmq -> .lml conversion")
            with LosslessWriter(dst, channels=hdr.channels, rate=hdr.sample_rate,
                                window_samples=hdr.window_samples) as writer:
                for window in reader:
                    signal = decoder(window.payload)
                    writer.write_window(signal, timestamp_us=window.timestamp_us)

        elif src_mode == 'lossless' and dst.endswith('.lmq'):
            if encoder is None:
                raise ValueError("encoder required for .lml -> .lmq conversion")
            with NeuralWriter(dst, channels=hdr.channels, rate=hdr.sample_rate,
                              window_samples=hdr.window_samples) as writer:
                for window in reader:
                    signal = window.decode()
                    payload = encoder(signal)
                    writer.write_window(payload, timestamp_us=window.timestamp_us)

        else:
            raise ValueError(f"Cannot convert {src_mode} -> {dst}")


# ============================================================
# Decoder / Encoder wrappers
# ============================================================

class Decoder:
    """GPU-side decoder for .lmq neural windows.

    Wraps the Vocos decoder tiers. Accepts a Window from an LMQReader
    and returns the reconstructed [channels, samples] signal.

    Usage:
        decoder = lq.Decoder(tier=5)
        with lq.open("session.lmq") as r:
            for window in r:
                signal = decoder.decode(window)  # [21, 2500]
    """

    # Tier configs match run_decoder_tier.py
    TIER_CONFIGS = {
        5: {'dim': 896,  'blocks': 20, 'exp': 3, 'n_fft': 32,  'params': '~100M'},
        6: {'dim': 1792, 'blocks': 20, 'exp': 3, 'n_fft': 64,  'params': '~400M'},
        7: {'dim': 1792, 'blocks': 32, 'exp': 4, 'n_fft': 128, 'params': '~837M'},
    }

    def __init__(self, tier: int = 5, checkpoint: Optional[str] = None,
                 device: str = 'cuda'):
        """Initialize decoder.

        Args:
            tier: Decoder tier (5, 6, or 7). Determines model size.
            checkpoint: Path to decoder checkpoint. If None, uses default.
            device: 'cuda' or 'cpu'.
        """
        if tier not in self.TIER_CONFIGS:
            raise ValueError(f"Unknown tier {tier}. Available: {list(self.TIER_CONFIGS)}")
        self.tier = tier
        self.device = device
        self.checkpoint = checkpoint
        self._model = None  # lazy load

    @property
    def config(self) -> dict:
        return self.TIER_CONFIGS[self.tier]

    def decode(self, window: Window) -> np.ndarray:
        """Decode a neural window to [channels, samples] signal.

        Args:
            window: Window from LMQReader (.lmq file)
        Returns:
            signal: [channels, samples] float64 array
        """
        if window.mode != 'neural':
            raise ValueError("Decoder.decode() only accepts neural windows. "
                             "Use window.decode() for lossless.")
        # Decompress the rANS payload to latent, then decode
        import torch
        from lamquant_codec.codec import SubbandCodec

        # The payload is a self-describing SubbandCodec packet
        # Try adaptive first (LMQ3), fall back to uniform (LMQ2)
        magic = window.payload[:4]
        if magic == b'LMQ3':
            # TODO: full decode chain with inverse lifting
            pass
        # For now, decompress latent and return L3-level reconstruction
        # Full [21, 2500] reconstruction requires the inverse lifting chain
        raise NotImplementedError(
            f"Full decode chain not yet wired. Tier {self.tier} model loading "
            f"and inverse lifting integration is pending."
        )

    def __repr__(self):
        cfg = self.config
        return f"Decoder(tier={self.tier}, {cfg['params']}, device={self.device!r})"


class Encoder:
    """Encoder wrapper for producing .lmq payloads from raw EEG.

    Drives the typed pipeline: decompose → encode (with SNN) → compress.

    Adaptive FSQ (LMQ3) is the default. The Mamba SNN classifies each
    timestep and produces a per-timestep FSQ level schedule, which the
    typed compress() path emits as an LMQ3 packet. Set `adaptive=False`
    (or `--no-adaptive-fsq` at the CLI) to force the uniform LMQ1 path.

    SNN resolution order:
      1. Explicit `snn_checkpoint=...` kwarg (e.g. for canary testing)
      2. PCCP registry `models.snn.production_checkpoint` (production)
      3. None → raise AdaptiveFSQError unless `adaptive=False`

    Usage:
        encoder = lq.Encoder(checkpoint="weights/student_subband.ckpt")
        with lq.NeuralWriter("session.lmq") as w:
            for segment in segments:
                payload, levels = encoder.encode(segment)
                # NeuralWriter auto-detects LMQ3 magic and sets the
                # FLAG_ADAPTIVE_FSQ bit; pass levels=b'\\x00'*10 for
                # adaptive (canonical) or actual bytes for legacy.
                w.write_window(payload, fsq_levels=levels)
    """

    def __init__(self, checkpoint: Optional[str] = None, quality: int = 2,
                 device: str = 'cpu', *,
                 snn_checkpoint=None,
                 adaptive: bool = True):
        self.checkpoint = checkpoint
        self.quality = quality
        self.device = device
        self.snn_checkpoint = snn_checkpoint
        self.adaptive = adaptive
        self._codec = None

    def _ensure_codec(self):
        if self._codec is not None:
            return
        from lamquant_codec.codec import SubbandCodec
        codec = SubbandCodec.from_checkpoint(self.checkpoint)

        if self.adaptive:
            from lamquant_codec.models.snn import (
                load_mamba_snn, resolve_production_snn,
            )
            from lamquant_codec.errors import AdaptiveFSQError

            snn_path = self.snn_checkpoint
            if snn_path is None:
                snn_path = resolve_production_snn()
            if snn_path is None:
                raise AdaptiveFSQError(
                    "Adaptive FSQ is enabled but no SNN is available: "
                    "registry pin is placeholder and no --snn-checkpoint "
                    "was provided. Either capture the production SHA via "
                    "`pccp_gate.py --capture --model snn --candidate <path>`, "
                    "pass `snn_checkpoint=<path>`, or set "
                    "`adaptive=False` (`--no-adaptive-fsq` at the CLI)."
                )
            snn = load_mamba_snn(snn_path, device=self.device)
            codec.set_snn(snn)

        self._codec = codec

    def encode(self, signal: np.ndarray) -> tuple:
        """Encode raw EEG to compressed payload.

        Args:
            signal: [channels, samples] EEG signal (float or int)
        Returns:
            (payload_bytes, fsq_levels_bytes) — for adaptive (LMQ3) the
            levels bytes are b'\\x00'*10 (canonical: payload owns the
            schedule). For uniform (LMQ1), the bytes are b'\\x02'*10
            placeholder (decoder reads L from packet header).
        """
        from lamquant_codec.codec_types import RawEEG
        from lamquant_codec.decompose import decompose
        from lamquant_codec.encode import encode as typed_encode
        from lamquant_codec.compress import compress as typed_compress

        self._ensure_codec()

        raw = RawEEG(signal=signal.astype(np.float32))
        sub = decompose(raw)
        tokens = typed_encode(sub, self._codec.model, snn=self._codec.snn)
        packet = typed_compress(tokens, sub, quality_mode=self.quality)

        # Defensive: assert adaptive/magic consistency. A codec
        # regression that emits LMQ3 with adaptive=False would silently
        # produce a file with mismatched flag + header zeroing, which
        # would break LMQ1-only readers. Fail loudly instead.
        is_lmq3 = packet.data[:4] == LMQ3_MAGIC
        if is_lmq3 and not self.adaptive:
            from lamquant_codec.errors import AdaptiveFSQError
            raise AdaptiveFSQError(
                "codec produced an LMQ3 payload while adaptive=False — "
                "this indicates a regression in typed_compress() or in "
                "Encoder.set_snn() routing. Refusing to emit a "
                "format-flag-desynced .lmq file."
            )
        if self.adaptive and is_lmq3:
            return packet.data, b'\x00' * 10  # canonical zero
        return packet.data, b'\x02' * 10      # legacy placeholder

    def __repr__(self):
        return (f"Encoder(quality={self.quality}, device={self.device!r}, "
                f"adaptive={self.adaptive})")


def info(path: str) -> dict:
    """Inspect a .lmq or .lml file without reading all windows.

    Returns a dict with file metadata, window count, duration, etc.

    Usage:
        >>> lq.info("recording.lml")
        {'format': 'lossless', 'magic': 'LQL1', 'channels': 21,
         'sample_rate': 250, 'windows': 120, 'duration_s': 1200.0, ...}
    """
    import os
    with open_file(path) as r:
        hdr = r.file_header
        n = r.window_count
        file_size = os.path.getsize(path)

        duration_s = None
        if n > 0 and r._index:
            first_ts = r._index[0].timestamp_us
            last_ts = r._index[-1].timestamp_us
            duration_s = (last_ts - first_ts) / 1_000_000.0
            duration_s += hdr.window_samples / hdr.sample_rate  # add last window

        return {
            'path': path,
            'format': r.mode,
            'magic': hdr.magic.decode('ascii'),
            'version': hdr.version,
            'channels': hdr.channels,
            'sample_rate': hdr.sample_rate,
            'window_samples': hdr.window_samples,
            'session_id': hdr.session_id,
            'decoder_tier_hint': hdr.decoder_tier_hint,
            'windows': n,
            'duration_s': duration_s,
            'file_size_bytes': file_size,
        }
