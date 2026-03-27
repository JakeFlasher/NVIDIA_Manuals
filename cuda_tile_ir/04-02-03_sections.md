---
title: "4.2.3. Sections"
section: "4.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#sections"
---

### [4.2.3. Sections](https://docs.nvidia.com/cuda/tile-ir/latest/sections#sections)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sections "Permalink to this headline")

The remainder of the bytecode stream is composed of sections.  Sections are used to group data within the
bytecode and allow operations on the stream to be per-section enabling out-of-order processing of data and/or lazy-loading.
Each section contains a section ID, whose high bit indicates if the section has alignment requirements, a
length and an optional alignment. A section ID is a 7-bit integer with the high bit indicating if the section
has alignment requirements. When an alignment is present, a variable number of padding bytes (each byte = `0xCB`)
may appear before the section data. The alignment of a section must be a power of 2. The alignment is represented
as a `VarInt`. The padding ensures the section data starts at the specified alignment boundary.

```text
section {
  idAndIsAligned: byte   // low 7 bits = section ID, high bit = alignment bit
  length: varint,
  alignment: varint?,    // present only if high bit was set
  padding: byte[],       // bytes of 0xCB as needed
  data: byte[]
}
```
