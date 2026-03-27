---
title: "5.1.3. Floating-Point Conversion Semantics"
section: "5.1.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#floating-point-conversion-semantics"
---

### [5.1.3. Floating-Point Conversion Semantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections#floating-point-conversion-semantics)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#floating-point-conversion-semantics "Permalink to this headline")

When converting values to a floating-point type (via [cuda_tile.ftof](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-ftof) or [cuda_tile.itof](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-itof)),
the behavior for out-of-finite-range values and special values depends on the target type.

**f16, f32, f64** (IEEE types) and **bf16, tf32** (IEEE-like types): the closest representable value is selected
according to the specified rounding mode. This may produce `Inf` when the source value exceeds
the target’s finite range. `NaN` values are preserved.

**e4m3, e5m2** (low-precision float types): use saturation-to-finite (satfinite) semantics, meaning the closest
representable _finite_ value is selected according to the specified rounding mode. `Inf` is
never produced, even if the source value was `Inf`.

Table [Floating-Point Conversion: Special Value and Saturation Behavior](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#table-float-conversion-semantics) enumerates the behavior for the different supported floating-point types
when converting various “corner case” values.

| Target Type | Out-of-Range Finite Source | Source is ±Inf | Source is NaN |
| --- | --- | --- | --- |
| f16, f32, f64, bf16, tf32 | Nearest representable value (may produce Inf) | ±Inf | NaN |
| e5m2 | Nearest representable finite value (±MAX_NORM) | ±MAX_NORM | NaN |
| e4m3 | Nearest representable finite value (±MAX_NORM) | ±MAX_NORM | +MAX_NORM |

> **Note**
>
> The e4m3 type does not support `NaN`; `NaN` inputs are converted to positive MAX_NORM.
