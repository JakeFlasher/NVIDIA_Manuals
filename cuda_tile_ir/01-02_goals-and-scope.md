---
title: "1.2. Goals and Scope"
section: "1.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html#goals-and-scope"
---

## [1.2. Goals and Scope](https://docs.nvidia.com/cuda/tile-ir/latest/sections#goals-and-scope)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#goals-and-scope "Permalink to this headline")

**Tile IR** aims to provide a performance-portable programming model and instruction set for tile-based parallel programming.

Core goals:

- Introduce a data-parallel tile programming abstraction that aligns with programmer intent and acts as a code generation target for DSLs and compilers.
- Abstract tensor-cores and their programming model to enable hardware innovation without disrupting the NVIDIA ecosystem or customers’ existing software investments.
- Abstract low-level, architecture-specific details such as CUDA threads and memory hierarchy to ensure programs can be efficiently recompiled for different GPUs.
- Minimize abstraction overhead. While portability has a cost, **Tile IR** aims to keep this cost low, introducing only a modest performance overhead.
- Provide user controls and optimization hints to recover peak performance when needed.
- Provide seamless interoperability with existing CUDA programming models like CUDA C++ and PTX.

**Tile IR** achieves these core goals through the following key components, in order of importance:

1. A versioned specification of the **Tile IR** abstract machine, including a versioned and portable bytecode representation.
2. An optimizing compiler for **Tile IR** programs, available as part of the CUDA driver and as a standalone tool in the CUDA toolkit, similar to PTX.
3. An MLIR dialect that existing compilers can use to target **Tile IR** as a backend compiler target.
