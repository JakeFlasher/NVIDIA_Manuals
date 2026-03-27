---
title: "Debug Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#debug-section"
---

#### [Debug Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#debug-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#debug-section "Permalink to this headline")

The debug section stores the serialized debug information (for more details about debug information see [Debug Info](https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#section-debug-info)).
This section is optional as certain tools may ignore it and serializers may leave it empty for release builds.

```text
debug {
  diOpsNum: varint          // Total number of operations with debug info
  padding: bytes            // Align to 4 bytes
  diIndexOffsets: uint32_t[] // Per op offset into the debug info indices
  diIndicesNum: varint      // Total number of debug info indices
  padding: bytes            // Align to 4 bytes
  diIndices: uint64_t[]     // Array of debug indices to debug info attributes
  diAttrNum: varint         // Total number of debug info attributes
  padding: bytes            // Align to 4 bytes
  diOffsets: uint32_t[]     // Per debug info attribute offset into the debug info data
  diData: bytes             // Data for each debug info attribute
}
```

The debug section uses a multi-level indirection scheme:

1. **Operations**: `diOpsNum` operations have debug info, with `diIndexOffsets` pointing into the indices array
2. **Indices**: `diIndicesNum` total indices in `diIndices`, referencing debug info attributes
3. **Attributes**: `diAttrNum` debug info attributes stored in `diData` with offsets in `diOffsets`

Each debug info attribute begins with a `debugEntryType` indicating what kind of debug info it is:

- `0x00` = “Unknown”
- `0x01` = “DICompileUnit”
- `0x02` = “DIFile”
- `0x03` = “DILexicalBlock”
- `0x04` = “DILoc”
- `0x05` = “DISubprogram”
- `0x06` = “CallSite”

`debugEntryPayload` describes line, file, variable name, function index, instruction
offset, etc. Each `debugEntryType` has a fixed, known structure. If
`functionLocIndex[i]` or an instruction’s `locationIndex` is non-zero, it references
`debugEntryOffset[...]` in this section, whose payload can store file/line info or
other metadata.
