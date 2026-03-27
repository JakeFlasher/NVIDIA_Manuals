---
title: "8.5. Control Flow"
section: "8.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#control-flow"
---

## [8.5. Control Flow](https://docs.nvidia.com/cuda/tile-ir/latest/sections#control-flow)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#control-flow "Permalink to this headline")

**Tile IR** contains a standard set of control flow operations that enable conditionals, and loops.

The operations are designed in the style of the [MLIR Control Flow dialect](https://mlir.llvm.org/docs/Dialects/ControlFlow/).

A notable difference is that we allow the nesting of control flow operations for example a [cuda_tile.if](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-if) may appear
inside a [cuda_tile.loop](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-loop) or [cuda_tile.for](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-for).

The main control structures are:

- [cuda_tile.if](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-if) which implements conditional branching.
- [cuda_tile.loop](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-loop) which implements a loop with arbitrary exit conditions.
- [cuda_tile.for](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-for) which implements a range-based loop with a fixed number of iterations.

These operations and their supporting operations are described in the following section.
