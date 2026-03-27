---
title: "SameElements"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#sameelements"
---

#### [SameElements](https://docs.nvidia.com/cuda/tile-ir/latest/sections#sameelements)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sameelements "Permalink to this headline")

```mlir
#same_elements< $values >
```

The `same_elements` attribute must be used as a predicate for
`cuda_tile.assume`. The predicated value must be a tensor of integers or
pointers.

`same_elements` is specified for each dimension. A value of C for a
dimension of size N indicates that, after dividing the respective
dimension into N/C groups of size C, each group consists of the same
elements. As N/C may not divide evenly, the last group may have fewer
than C elements.

If the “same elements” property does not hold along a dimension, the
respective value should be set to 1.
`#cuda_tile.same_elements<[1, 1, ..., 1]>` is a correct predicate for any
tensor of integers or pointers, where the number of ones matches the rank
of the tensor. (Size-1 groups always have the same elements.)

```mlir
// Integer tensor with same elements.
%0 = cuda_tile.constant <i16: [[0, 0, 0, 0, 10, 10, 10, 10],
                               [0, 0, 0, 0, 10, 10, 10, 10],
                               [5, 5, 5, 5, 93, 93, 93, 93],
                               [5, 5, 5, 5, 93, 93, 93, 93]]>
    : tile<4x8xi16>
%1 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %0
    : !cuda_tile.tile<4x8xi16>

// Pointer tensor with same elements.
%2 = cuda_tile.constant <i64: [[ 0,  0,  0,  0,  8,  8,  8,  8],
                               [ 0,  0,  0,  0,  8,  8,  8,  8],
                               [64, 64, 64, 64, 32, 32, 32, 32],
                               [64, 64, 64, 64, 32, 32, 32, 32]]>
    : tile<4x8xi64>
%3 = cuda_tile.bitcast %2
    : !cuda_tile.tile<4x8xi64>
      -> !cuda_tile.tile<!cuda_tile.ptr<f32>>
%4 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %3
    : !cuda_tile.tile<!cuda_tile.ptr<f32>>
```
