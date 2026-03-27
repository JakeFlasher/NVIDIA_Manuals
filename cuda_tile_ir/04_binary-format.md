---
title: "4. Binary Format"
section: "4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#binary-format"
---

# [4. Binary Format](https://docs.nvidia.com/cuda/tile-ir/latest/sections#binary-format)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#binary-format "Permalink to this headline")

The **Tile IR** bytecode is a binary representation of **Tile IR** modules including
all items (globals, kernels, functions), attributes, and instructions.

The bytecode format is stable, versioned, and provides **Tile IR** portability guarantees.
The bytecode format produced by older **Tile IR** compilers/drivers can be read by newer compilers/drivers.
A compiler/driver accepts bytecode up to its supported **Tile IR** version.

The remainder of this section describes the **Tile IR** bytecode format and its encoding.

The bytecode has a few top-level design goals:
- To represent a finite but expandable over time set of operations.
- To use a minimal set of types for the encoding.
- To enable manual construction and inspection by humans.
- To allow lazy loading of functions.
- To support forward and backward compatibility of the encoding.

The encoding of individual operations is described in [Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#section-operations).
