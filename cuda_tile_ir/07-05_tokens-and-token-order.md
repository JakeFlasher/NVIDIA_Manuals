---
title: "7.5. Tokens and token order"
section: "7.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#tokens-and-token-order"
---

## [7.5. Tokens and token order](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tokens-and-token-order)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tokens-and-token-order "Permalink to this headline")

In **Tile IR** we have explicit annotation of dependencies between loads and stores for the token-ordered operations.
**Tile IR** produces wide loads and stores of whole tiles of data, making efficient use of various resources of the GPU in parallel.
We provide token ordered operations to explicitly inform the **Tile IR** toolchain that two operations may happen in parallel, and will not interfere with each other.
There is a family of memory operations called token ordered operations which produce and consume tokens.
Tokens are abstract values in the **Tile IR** language for building dependencies between memory operations within the same tile block thread.
They have no concrete representation at runtime, cannot be compared, computed upon, or stored/loaded to/from memory.

Program dependencies (i.e. dependencies apparent from control flow, data dependency, or address dependency) **do not** provide ordering between two memory operations.
Tokens must be used, even where the token ordering appears redundant with program dependencies.
Program dependencies may be optimized away by the **Tile IR** toolchain, whereas token dependencies are not.
