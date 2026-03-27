---
title: "5.1.1. Fundamental Types"
section: "5.1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#fundamental-types"
---

### [5.1.1. Fundamental Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#fundamental-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#fundamental-types "Permalink to this headline")

**Tile IR** supports a set of general purpose integer and floating-point
types that are supported by all operations and have no restrictions.
These can be contained in arbitrary rank, and shape tensors, and 0-rank
values of this type can be treated as scalars.

| Type | Sizes | Description |
| --- | --- | --- |
| i1, i8, i16, i32, i64 | 1, 8, 16, 32, 64 | signless integer type of specified size |
| f16, f32, f64 | 16, 32, 64 | IEEE floating-point type of specified size |

Primary elemental types are supported in all arithmetic operations.

> **Warning**
>
> Integer types are signless, i.e., the type does not encode whether
> the represented value is to be interpreted as a signed or unsigned
> value. For operations where this distinction is semantically
> meaningful signedness is controlled via flags on each arithmetic
> operation.
