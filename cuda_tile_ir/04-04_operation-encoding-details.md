---
title: "4.4. Operation Encoding Details"
section: "4.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#operation-encoding-details"
---

## [4.4. Operation Encoding Details](https://docs.nvidia.com/cuda/tile-ir/latest/sections#operation-encoding-details)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#operation-encoding-details "Permalink to this headline")

The general structure for operation encoding follows a consistent pattern, but varies based on the operation’s characteristics:

**General Operation Structure**

```text
opcode : byte                     // Operation identifier
locationIndex : varint            // Debug location (0 = no debug info)
resultTypes : typeIndex[]?        // Present for variadic result operations
flags : varint?                   // Optional flags for operations with optional fields
attributes : encoded_attr[]       // Operation-specific attributes
operands : operand_encoding       // Operation-specific operand encoding
regions : region_encoding[]?      // Present for operations with regions
```

**Flags Field Encoding**

For operations with optional attributes or operands, a flags field is used:

```text
flags : varint                    // Bitfield encoding optional presence
```

The flags field uses individual bits to indicate the presence of optional attributes and operands:

- **Bits 0-N**: Optional attributes (in declaration order)
- **Bits N+1-M**: Optional operands (in declaration order)

**UnitAttr** attributes are encoded only in the flags field - no additional data is written.

**Operand Encoding Patterns**

Operands are encoded differently based on the operation’s operand structure:

1. **Fixed Operands**: Written as sequential operand indices
2. **Variadic Operands**: Prefixed with operand count, then indices
3. **AttrSizedOperandSegments**: Each operand group encoded separately

```text
// Fixed operands (e.g., binary operations)
operand1Index : varint
operand2Index : varint

// Variadic operands (e.g., function calls)
numOperands : varint
operandIndices : varint[numOperands]

// AttrSizedOperandSegments (e.g., operations with optional operand groups)
group1 : optional_operand_group
group2 : optional_operand_group
...
```

**Result Type Encoding**

- **Fixed Results**: No result types encoded (inferred from operation)
- **Variadic Results**: Number of results encoded, followed by type indices

```text
// Variadic results
numResults : varint
resultTypeIndices : varint[numResults]
```

**Region Encoding**

Operations with regions encode them after operands:

```text
numRegions : varint
regions : region[numRegions]

region {
  numBlocks : varint
  blocks : block[numBlocks]
}

block {
  numArgs : varint
  argTypeIndices : varint[numArgs]
  numOps : varint
  operations : operation[numOps]
}
```

**Common Operation Examples**

**Arithmetic Operations** (e.g., `cuda_tile.add`)

```text
opcode : byte                     // e.g., 0x15 for add
locationIndex : varint            // Debug location
lhs : varint                      // Left operand index
rhs : varint                      // Right operand index
// Result type inferred from operands
```

**Memory Operations** (e.g., `cuda_tile.load`)

```text
opcode : byte                     // e.g., 0x20 for load
locationIndex : varint            // Debug location
resultType : varint               // Type index for loaded value
address : varint                  // Address operand index
// Optional attributes encoded via flags
```

**Control Flow Operations** (e.g., `cuda_tile.if`)

```text
opcode : byte                     // e.g., 0x30 for if
locationIndex : varint            // Debug location
condition : varint                // Condition operand index
numRegions : varint               // Always 2 for if (then, else)
thenRegion : region               // Then block
     elseRegion : region               // Else block (may be empty)
```
