---
title: "8.7.2. Floating-Point Math"
section: "8.7.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#floating-point-math"
---

### [8.7.2. Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections#floating-point-math)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#floating-point-math "Permalink to this headline")

**Tile IR** contains a set of standard math library operations which implement familiar mathematical functions over tensors
supporting 16-bit, 32-bit, and 64-bit floating-point data types.

> **Note**
>
> 32-bit and 64-bit operations typically leverage efficient hardware-specific
> instructions. Some 16-bit operations are emulated using wider intermediate
> computations, and may not offer the same performance.

> **Warning**
>
> There are some restrictions based on data type support which are detailed in the [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types) section.
