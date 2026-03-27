---
title: "Attribute Encoding"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#attribute-encoding"
---

#### [Attribute Encoding](https://docs.nvidia.com/cuda/tile-ir/latest/sections#attribute-encoding)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#attribute-encoding "Permalink to this headline")

Attributes in **Tile IR** bytecode can be encoded in two ways:

1. **Inline encoding** - Simple attributes are encoded directly in the instruction stream
2. **Self-contained encoding** - Complex attributes include a type tag followed by their data

Self-contained attributes use the following format:

```text
attributeTag : byte
attributeData : byte[]  // Format depends on attributeTag
```

The following attribute tags are supported:

**Integer Attribute** (`attributeTag` = 0x01)

```text
attributeTag : byte = 0x01  // Integer attribute
typeIndex : varint          // Type index for the integer type
value : varint              // Integer value (zero-extended)
```

**Float Attribute** (`attributeTag` = 0x02)

```text
attributeTag : byte = 0x02  // Float attribute
typeIndex : varint          // Type index for the float type
value : byte[]              // APFloat representation (variable length)
```

**Bool Attribute** (`attributeTag` = 0x03)

```text
attributeTag : byte = 0x03  // Bool attribute
value : byte                // 0x00=false, 0x01=true
```

**Type Attribute** (`attributeTag` = 0x04)

```text
attributeTag : byte = 0x04  // Type attribute
typeIndex : varint          // Index of the referenced type
```

**String Attribute** (`attributeTag` = 0x05)

```text
attributeTag : byte = 0x05  // String attribute
stringIndex : varint        // Index into the string section
```

**Array Attribute** (`attributeTag` = 0x06)

```text
attributeTag : byte = 0x06    // Array attribute
numElements : varint          // Number of elements
elements : self-contained[]   // Array of self-contained attributes
```

**DenseElements Attribute** (`attributeTag` = 0x07)

```text
attributeTag : byte = 0x07  // DenseElements attribute
typeIndex : varint          // Type index for the shaped type
constantIndex : varint      // Index into constant section (for numeric data)
// OR for string data:
numStrings : varint         // Number of string elements
stringIndices : varint[]    // Indices into string section
```

**DivBy Attribute** (`attributeTag` = 0x08)

```text
attributeTag : byte = 0x08    // DivBy attribute
divisor : varint              // Divisor value
flags : byte                  // Bit 0: unsignedInt, Bit 1: hasEvery, Bit 2: hasAlong
every : signed_varint?        // Present if Bit 1 set
along : signed_varint?        // Present if Bit 2 set
```

**SameElements Attribute** (`attributeTag` = 0x09)

```text
attributeTag : byte = 0x09  // SameElements attribute
values : int64_t[]          // Array of int64 values (var-length)
```

**Dictionary Attribute** (`attributeTag` = 0x0A)

```text
attributeTag : byte = 0x0A      // Dictionary attribute
numEntries : varint             // Number of key-value pairs
entries : dictEntry[]           // Array of dictionary entries

dictEntry {
  keyStringIndex : varint       // Index of key string
  value : self-contained        // Self-contained attribute value
}
```

**OptimizationHints Attribute** (`attributeTag` = 0x0B)

```text
attributeTag : byte = 0x0B  // OptimizationHints attribute
dictionary : dictionary    // Dictionary attribute (without tag)
```

**NonNegative Attribute** (`attributeTag` = 0x0C)

```text
attributeTag : byte = 0x0C  // NonNegative attribute
// No additional payload - presence indicates non-negative constraint
```
