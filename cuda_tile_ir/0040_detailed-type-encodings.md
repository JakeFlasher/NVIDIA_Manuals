---
title: "Detailed Type Encodings"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#detailed-type-encodings"
---

#### [Detailed Type Encodings](https://docs.nvidia.com/cuda/tile-ir/latest/sections#detailed-type-encodings)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#detailed-type-encodings "Permalink to this headline")

The following sections describe the specific encoding format for each type tag:

**Integer Types** (`typeTag` = 0x00-0x04)

Integer types require no additional payload data beyond the type tag:

```text
I1:   typeTag = 0x00  // 1-bit boolean
I8:   typeTag = 0x01  // 8-bit integer
I16:  typeTag = 0x02  // 16-bit integer
I32:  typeTag = 0x03  // 32-bit integer
I64:  typeTag = 0x04  // 64-bit integer
```

**Float Types** (`typeTag` = 0x05-0x0B)

Float types require no additional payload data beyond the type tag:

```text
F16:      typeTag = 0x05  // 16-bit IEEE float
BF16:     typeTag = 0x06  // 16-bit bfloat
F32:      typeTag = 0x07  // 32-bit IEEE float
TF32:     typeTag = 0x08  // TensorFloat-32
F64:      typeTag = 0x09  // 64-bit IEEE float
F8E4M3FN: typeTag = 0x0A  // 8-bit float (E4M3FN format)
F8E5M2:   typeTag = 0x0B  // 8-bit float (E5M2 format)
```

**Pointer Type** (`typeTag` = 0x0C)

```text
typeTag : byte = 0x0C      // Pointer type
pointeeTypeIndex : varint  // Index of the pointee type
```

**Tile Type** (`typeTag` = 0x0D)

```text
typeTag : byte = 0x0D      // Tile type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
```

**TensorView Type** (`typeTag` = 0x0E)

```text
typeTag : byte = 0x0E      // TensorView type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
strides : int64_t[]        // Stride values (var-length array)
indexTypeTag : byte        // Index type (I32=0x03 or I64=0x04)
```

**PartitionView Type** (`typeTag` = 0x0F)

```text
typeTag : byte = 0x0F        // PartitionView type
tileShape : int32_t[]        // Tile shape (var-length array)
tensorViewTypeIndex : varint // Index of the TensorView type
dimMap : int32_t[]           // Dimension mapping (var-length array)
masked : byte                // Masked flag (0x00=false, 0x01=true)
```

**Function Type** (`typeTag` = 0x10)

```text
typeTag : byte = 0x10 // Function Type
numParams : varint    // Number of input parameters
paramTypeIndices : varint[]  // Array of type indices for each parameter
numResults : varint   // Number of results
resultTypeIndices : varint[] // Array of type indices for each return value
```

The function type encoding stores only the essential type information (input and result types).
Argument attributes and other function metadata are stored separately in the function table section.

**Token Type** (`typeTag` = 0x11)

```text
typeTag : byte = 0x11  // Token type (no additional payload)
```
