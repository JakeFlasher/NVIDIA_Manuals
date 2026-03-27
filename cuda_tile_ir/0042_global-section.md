---
title: "Global Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#global-section"
---

#### [Global Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#global-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#global-section "Permalink to this headline")

The global section stores module-level global variables. This section is optional and is only present if the module contains `cuda_tile.global` operations.

```text
global {
  numGlobals: varint
  // for each global i in [0..numGlobals-1]:
  symbolNameIndex[i] : varint   // References the global's symbol name in the StringSec
  valueTypeIndex[i] : varint    // Type index for the global's value type
  constantValueIndex[i] : varint // Index into constant section for the global's initial value
}
```

Each global variable is encoded with:

- **symbolNameIndex**: Index into the string section for the global’s symbol name
- **valueTypeIndex**: Index into the type section for the global’s type (typically a shaped type like tensor)
- **constantValueIndex**: Index into the constant section containing the global’s initial value
