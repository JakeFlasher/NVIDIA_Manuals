---
title: "4.1.1. Fixed-Width Integers"
section: "4.1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#fixed-width-integers"
---

### [4.1.1. Fixed-Width Integers](https://docs.nvidia.com/cuda/tile-ir/latest/sections#fixed-width-integers)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#fixed-width-integers "Permalink to this headline")

Fixed width integers are unsigned integers of a known size (in bytes). The values are
stored in little-endian byte order.

**Note**: All multi-byte values in the **Tile IR** bytecode format use little-endian encoding,
including fixed-width integers, array offsets, and type indices.

```text
byte ::= `0x00`...`0xFF`
```
