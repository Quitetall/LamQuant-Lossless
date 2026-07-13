# BGF1 body format v1 (draft)

**Status:** DRAFT. Native modes are implemented and tested. Learned graph/model
sections remain reserved; this document must not be marked frozen until the
cross-language learned packet golden passes ADR 0116.

An Optimum-v2 packet begins with the seven-byte `LMO1` envelope:

```
offset size field
0      4    magic = "LMO1"
4      1    wire_version = 3
5      1    mode = 0 (lossless)
6      1    body = 3 (BGF1)
```

The 80-byte BGF1 header immediately follows:

```
offset size field
0      4    magic = "BGF1"
4      1    version = 1
5      1    flags = 0
6      1    bit_depth (1..32)
7      1    reserved = 0
8      2    channel_count (u16 LE, 1..256)
10     2    tile_count (u16 LE; native v1 = 1)
12     4    samples_per_channel (u32 LE, 1..32768)
16     4    sample_rate_millihertz (u32 LE)
20     4    model_id (native = 0)
24     32   model_sha256 (native = all zero)
56     4    labels_length (u32 LE)
60     4    graph_length (u32 LE; native = 0)
64     4    tile_directory_length (u32 LE)
68     4    payload_length (u32 LE)
72     4    decoded_i32_crc32c (u32 LE)
76     4    packet_crc32c (u32 LE)
```

Packet CRC32C is Castagnoli CRC-32C over the complete `LMO1` envelope, BGF1
header with bytes 76..79 treated as zero, and every body byte. Decoded CRC32C
covers channel-major signed-i32 little-endian samples.

Body order is labels, graph, tile directory, payload. Labels are repeated
`u16 byte_length` followed by UTF-8 bytes, exactly one label per channel.

Each native tile directory entry is 24 bytes:

```
0  mode (0 = raw signed i32 LE, 1 = delta ZigZag canonical unsigned LEB128)
1  flags = 0
2  reserved u16 = 0
4  first_sample u32 = 0
8  sample_count u32 = samples_per_channel
12 payload_offset u32 = 0
16 payload_length u32
20 decoded_i32_crc32c u32
```

The native v1 directory must cover the complete window exactly. Delta symbols
are channel-major; each channel resets its previous value to zero. Overlong
LEB128 encodings, gaps, overlaps, trailing bytes, nonzero reserved fields,
dimension products above 8,388,608 values, and samples outside declared bit
depth are invalid.
