---
title: "4.1.2. Variable-Width Integers"
section: "4.1.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#variable-width-integers"
---

### [4.1.2. Variable-Width Integers](https://docs.nvidia.com/cuda/tile-ir/latest/sections#variable-width-integers)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#variable-width-integers "Permalink to this headline")

Variable width integers, or `VarInt`s, provide a compact representation for integers.
Each encoded `VarInt` consists of one to nine bytes, which together represent a single
64-bit value. The **Tile IR** bytecode utilizes the `PrefixVarInt` encoding for `VarInt`s.
This encoding is a variant of the `LEB128` (“Little-Endian Base 128”) encoding, where each
byte of the encoding provides up to `7` bits for the value, with the remaining bit used to
store a tag indicating the number of bytes used for the encoding. Small unsigned integers
(less than `2^7`) may be stored in one byte, larger unsigned integers (up to `2^14`) may
be stored in two bytes, and so on.

```text
varint ::= `0x00`...`0xFF`
```
