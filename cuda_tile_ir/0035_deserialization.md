---
title: "Deserialization"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#deserialization"
---

#### [Deserialization](https://docs.nvidia.com/cuda/tile-ir/latest/sections#deserialization)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#deserialization "Permalink to this headline")

The following are the implications of the section format on the deserialization process.

1. The deserializer must read the section ID and length (`idAndIsAligned`) as a single byte.
2. If the section ID has the high bit set, the deserializer must read the alignment (`alignment`) and apply the alignment by skipping
`0xCB` bytes as needed.
3. The deserializer must read the length (`length`) bytes of the payload. The length is a `VarInt`.
4. The deserializer must move on to the next section until `EOF` (end of file).
