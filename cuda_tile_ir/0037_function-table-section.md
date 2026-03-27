---
title: "Function Table Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#function-table-section"
---

#### [Function Table Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#function-table-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#function-table-section "Permalink to this headline")

The function table section enumerates the module’s functions and embeds their code inline. First,
`numFunctions` indicates how many functions follow; for each function `i`,
`nameIndex[i]` is an index into the string section naming the function and
`signatureIndex[i]` is an index into the Type section specifying its parameter, return
types, global flags, etc (so multiple functions with identical signatures can share the
same type entry). The field `functionLocIndex[i]` is a `VarInt` referencing an entry
in the Debug Section that describes the function’s definition location (e.g., source
file and line). If `functionLocIndex[i]` is zero, there is no associated debug
metadata for the function’s definition scope. `lengthOfFunction[i]` states how many
bytes of code belong to function `i`, and `functionBody[i]` contains exactly those
bytes of instruction encodings. This layout avoids a separate code section and makes
parsing each function straightforward: once you read the metadata for function `i`,
you can either parse its instructions directly or skip them by advancing
`lengthOfFunction[i]` bytes. The instruction encoding itself allows each operation to
include a slot for its source location, referencing an entry in the debug section.

The instruction encodings (opcodes, operands, etc.) are described later under Operation
Opcodes and Encodings Section.

The function table encoding has been updated to include function flags and optional optimization hints:

```text
functionTable {
  numFunctions : varint
  // for each function i in [0..numFunctions-1]:
  nameIndex[i] : varint                    // References the function's name in the StringSec
  signatureIndex[i] : varint               // References the extended function signature in the TypeSec
  entryFlag[i] : byte                      // Function flags (visibility, kind, optimization hints)
  functionLocIndex[i] : varint             // 0 means no debug location
  optimizationHints[i]? : self-contained  // Present only if HasOptimizationHints flag is set
  lengthOfFunction[i] : varint
  functionBody[i] : byte[lengthOfFunction[i]]
}
```

The `entryFlag` byte encodes the following information:

- **Bit 0 (0x01)**: Visibility Flag (0 = Public, 1 = Private)
- **Bit 1 (0x02)**: Function Kind Flag (0 = Device Function, 1 = Kernel Entry Point)
- **Bit 2 (0x04)**: Has Optimization Hints Flag (0 = No, 1 = Yes)
- **Bits 3-7**: Reserved for future extensions
