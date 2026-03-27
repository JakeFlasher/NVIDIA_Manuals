---
title: "Core CuTe DSL Abstractions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html#core-cute-dsl-abstractions"
---

## [Core CuTe DSL Abstractions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#core-cute-dsl-abstractions)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#core-cute-dsl-abstractions "Permalink to this headline")

- **Layouts** – Describe how data is organized in memory and across threads.
- **Tensors** – Combine data pointers or iterators with layout metadata.
- **Atoms** – Represent fundamental hardware operations like matrix multiply-accumulate (MMA) or memory copy.
- **Tiled Operations** – Define how atoms are applied across thread blocks and warps (e.g., `TiledMma`, `TiledCopy`).

For more on CuTe abstractions, refer to the [CuTe C++ library documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md).

**Pythonic Kernel Expression**

Developers express kernel logic, data movement, and computation using familiar Python syntax and control flow.

The DSLs simplify expressing loop tiling, threading strategies, and data transformations using concise Python code.

**JIT Compilation**

Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIA’s `ptxas` toolchain,
enabling rapid iteration and interactive debugging.
