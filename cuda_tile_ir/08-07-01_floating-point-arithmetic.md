---
title: "8.7.1. Floating-Point Arithmetic"
section: "8.7.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#floating-point-arithmetic"
---

### [8.7.1. Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections#floating-point-arithmetic)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#floating-point-arithmetic "Permalink to this headline")

Standard floating-point types implement the IEEE-754 standard for floating-point arithmetic. On NVIDIA hardware, certain types
are non-standard and _do not_ implement the IEEE-754 standard, see [Element Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) for more details about the
different floating-point types, their precision, storage, and formats.

Supports 16-bit, 32-bit, and 64-bit floating-point data types.
