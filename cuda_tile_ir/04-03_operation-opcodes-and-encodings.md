---
title: "4.3. Operation Opcodes and Encodings"
section: "4.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#operation-opcodes-and-encodings"
---

## [4.3. Operation Opcodes and Encodings](https://docs.nvidia.com/cuda/tile-ir/latest/sections#operation-opcodes-and-encodings)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#operation-opcodes-and-encodings "Permalink to this headline")

Each instruction in **Tile IR** bytecode is represented as:

```text
opcode : byte
locationIndex : varint
instructionSpecificFields : byte[]
```

In this representation, `opcode` uniquely identifies the operation. This matches one
of the operations defined in the **Tile IR** dialect. `locationIndex` is always
present; if the instruction does not carry debug info, this field is 0. A non-zero value
refers to an entry in the Debug Section that contains file, line, or other source-level
metadata. The instruction-specific fields (operands, attributes, etc.) follow a layout
defined by the opcode.

Additionally, we do not store a `resultIndex` for each producing instruction. Instead, we
rely on a sequential pass to assign local value indices at parse-time.

Older parsers are expected to skip or reject unknown opcodes. Over time, new operations
can be added simply by assigning new opcodes and defining their binary payload formats.
