---
title: "10. Stability"
section: "10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#stability"
---

# [10. Stability](https://docs.nvidia.com/cuda/tile-ir/latest/sections#stability)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#stability "Permalink to this headline")

**Tile IR** provides a set of guarantees regarding portability, stability, and compatibility to ensure predictable behavior across different
platforms, toolchains, and hardware targets. These guarantees are documented below.

Definitions:

- **Stability**: An unchanging property of a program or interface.
- **Portability**: A property of a program to be transfered to a different hardware or toolchain version with the same behavior.
- **Compatibility**: A property of a program to be executed on a different platform or toolchain with the same behavior.
- **Toolchain**: Either  the compiler and CTK version used to perform ahead of time compilation or, the driver and CTK version
used to perfomr JIT compilation of a **Tile IR** program.
