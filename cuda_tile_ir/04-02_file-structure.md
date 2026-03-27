---
title: "4.2. File Structure"
section: "4.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#file-structure"
---

## [4.2. File Structure](https://docs.nvidia.com/cuda/tile-ir/latest/sections#file-structure)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#file-structure "Permalink to this headline")

The **Tile IR** bytecode file is composed of a stream of bytes; which is composed of
a header followed by multiple sections. The header contains the 8-byte “magic number”,
and a version number.

Each section is expected to appear once within a bytecode file and is identified by a type code,
the length, alignment, padding, and the payload. This design allows forward compatibility and selective
parsing (tools can skip unknown sections, or unneeded sections).

```text
bytecode {
  magic: "\x7FTileIR\x00",
  version: varint,
  sections: section[]
}
```
