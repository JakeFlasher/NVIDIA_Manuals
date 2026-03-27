---
title: "8.2.2. Distinct Floating-Point and Integer Operations"
section: "8.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#distinct-floating-point-and-integer-operations"
---

### [8.2.2. Distinct Floating-Point and Integer Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections#distinct-floating-point-and-integer-operations)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#distinct-floating-point-and-integer-operations "Permalink to this headline")

Numeric ooerations are split across integer and floating-point types due to differences in flags such as rounding modes, `NaN` handling,
and fast math.

For example, the [cuda_tile.addf](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-addf) operation supports a *rounding* attribute, but the *addi* operation does not.
