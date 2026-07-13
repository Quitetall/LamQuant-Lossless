# LQW1 frozen integer model pack v1

**Status:** FROZEN for version 1. All integers are little-endian. Packs are at
most 64 MiB and contain 1..4096 tensors in strictly increasing UTF-8 name order.

The fixed 48-byte header is:

```
offset size field
0      4    magic = "LQW1"
4      1    version = 1
5      1    flags = 0
6      2    tensor_count u16
8      4    directory_length u32
12     4    payload_length u32
16     32   SHA-256
```

SHA-256 covers the complete pack with header bytes 16..47 treated as zero.

Each variable directory entry has a 20-byte fixed prefix followed by name and
shape:

```
0  name_length u16
2  dtype u8 (1=i8, 2=i16, 3=i32)
3  rank u8 (1..8)
4  scale_numerator i32
8  scale_shift u8 (0..31)
9  reserved[3] = 0
12 payload_offset u32
16 payload_length u32
20 name bytes, then rank x u32 dimensions
```

Tensor names are nonempty ASCII `[A-Za-z0-9_.-]+`; this avoids Unicode
normalization aliases. Tensor value scale is
`integer * scale_numerator / 2^scale_shift`. The numerator is nonzero and must
be odd whenever `scale_shift > 0`, so reducible power-of-two scale aliases are
invalid. Every shape
dimension is nonzero and the checked shape product times dtype width equals the
payload length. Payload ranges are canonical contiguous concatenation in
directory order: first offset zero, no gaps, no overlaps, no trailing bytes.
Duplicate or noncanonical names, invalid UTF-8, digest mismatch, or arithmetic
overflow are fatal.
