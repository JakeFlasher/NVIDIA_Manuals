---
title: "9. Debug Info"
section: "9"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#debug-info"
---

# [9. Debug Info](https://docs.nvidia.com/cuda/tile-ir/latest/sections#debug-info)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#debug-info "Permalink to this headline")

Debug info is essential for understanding and diagnosing the behavior of a **Tile IR** program. Debug info is metadata that maps generated code back to the original source code, creating a more informative and intuitive debugging experience for the **Tile IR** user and allowing the user to quickly identify and fix issues in a **Tile IR** program. This section explains how to include and use debug info in a **Tile IR** program when using the MLIR dialect. For a description of how to produce this information in the bytecode directly, see the [bytecode section](https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#section-bytecode).

**Tile IR** supports control flow based debugging of unoptimized code with features such as breakpoints, stepping through code and inspecting stack frames. **Tile IR** debug info exposes scope metadata describing the files, functions and blocks that comprise a **Tile IR** program along with information on how that program was compiled. This scope metadata is included in the **Tile IR** program by fusing it to location information. Details and examples below. **Tile IR** does not support value inspection of user variables, only control flow based debugging as described above is supported at this time.
