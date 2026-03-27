---
title: "11.2.27. cuda_tile.cmpf_0"
section: "11.2.27"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-cmpf-0"
---

### [11.2.27. cuda_tile.cmpf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cmpf-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cmpf-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example() {
     %lhs0 = constant <f16: 0.0> : tile<f16>
     %rhs0 = constant <f16: 0.0> : tile<f16>

     // Custom form of scalar "ordered equal" comparison.
     %x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

     %lhs1 = constant <f16: 0.0> : tile<2x2xf16>
     %rhs1 = constant <f16: 0.0> : tile<2x2xf16>

     // Custom form of scalar "unordered less than" comparison.
     %x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

     %lhs2 = constant <f64: 0.0> : tile<2x2xf64>
     %rhs2 = constant <f64: 0.0> : tile<2x2xf64>
  }
}
```
