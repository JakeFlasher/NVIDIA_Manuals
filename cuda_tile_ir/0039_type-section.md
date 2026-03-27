---
title: "Type Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#type-section"
---

#### [Type Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#type-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-section "Permalink to this headline")

The type section stores all type definitions (scalar types, function signatures, parametric
types, etc.) used in the module. The section begins with `numTypes` specifying how many types
follow, then an array of offsets (`typeStartIndex`) into the encoded blob (`typeData`). To load the
i-th type definition, you use the offset contained in `typeStartIndex[i]` to index into
the `typeData` blob and parse the definition from there. The `typeData` blob is a single encoded binary
blob containing the type definitions concatenated together.

```text
type {
  numTypes: varint
  typeStartIndex: uint32_t[] // array of offsets, length = numTypes
  typeData: byte[]           // concatenated bytes for all type definitions
}
```

Each individual type definition will consist of a `typeTag` followed by a predefined payload
structure specific to that `typeTag`. This deterministic approach allows parsers to
understand the `payload` layout based solely on the `typeTag`.

```text
typeTag : byte
payload : byte[lengthOfPayload]
```

- `typeTag` indicates the “kind” of type. This can be a simple scalar, a function
signature, or a more complex structure like a tensor.
- `payload` is interpreted based on `typeTag`. For instance, a function signature might
store the number of parameters, references to their types, etc, while a tensor type
might store dimensions and an element type.

Example `typeTag` Values:

| `typeTag`  | Meaning         | Expected Payload |  |  |
| --- | --- | --- |
| 0x01             \| i1 (1-bit int)  \| None |  |  |
| 0x02 | i32 (32-bit int) | None |
| 0x03 | i64 (64-bit int) | None |
| 0x04 | tensor | `elementTypeIndex + rank + dimensions[]` |
| 0x10 | Function | Extended function signature |
| … | Future types | Depends on the type |
