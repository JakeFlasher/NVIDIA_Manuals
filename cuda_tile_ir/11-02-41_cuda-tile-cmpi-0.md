---
title: "11.2.41. cuda_tile.cmpi_0"
section: "11.2.41"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-cmpi-0"
---

### [11.2.41. cuda_tile.cmpi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cmpi-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cmpi-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lhs0 = constant <i16: 0> : tile<i16>
     %rhs0 = constant <i16: 0> : tile<i16>

     // Scalar "signed less than" comparison.
     %x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

     %lhs1 = constant <i64: 0> : tile<2x2xi64>
     %rhs1 = constant <i64: 0> : tile<2x2xi64>

     // Tile equality comparison.
     // There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
     %x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
  }
}
```
