---
title: "4.5. Section Ordering"
section: "4.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#section-ordering"
---

## [4.5. Section Ordering](https://docs.nvidia.com/cuda/tile-ir/latest/sections#section-ordering)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#section-ordering "Permalink to this headline")

**Tile IR** bytecode readers can handle sections in any order due to their flexible parsing design.
The reader first discovers all sections and stores their payloads, then processes them in dependency order.

**Default Writing Order**

Writers typically emit sections in the following order:

1. **Header** (magic number + version)
2. **Global Section** (Section ID: 0x06) - Optional, only if globals present
3. **Function Table Section** (Section ID: 0x02) - Required
4. **Constant Data Section** (Section ID: 0x04) - Optional, only if constants present
5. **Debug Section** (Section ID: 0x03) - Optional
6. **Type Section** (Section ID: 0x05) - Required
7. **String Section** (Section ID: 0x01) - Required
8. **End-of-Bytecode Marker** (0x00) - Required

However, readers are not dependent on this order and can process sections regardless of their arrangement
in the file. This flexibility enables future optimizations and different writing strategies.

**Reader Implementation**

The reader implements this flexibility by:

1. **Discovery Phase**: Reads all section headers and stores their payloads in memory
2. **Processing Phase**: Processes sections in dependency order regardless of file order
3. **Lazy Resolution**: Resolves forward references (e.g., types, strings) on-demand

This design allows for efficient random access to any section and supports future file format optimizations.

**Section Dependencies**

- Function table references: types, strings, constants, debug info
- Global section references: types, strings, constants
- Debug section references: strings
- All sections may reference the string section
