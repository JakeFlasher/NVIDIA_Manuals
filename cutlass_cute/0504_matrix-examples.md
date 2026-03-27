---
title: "Matrix examples"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#matrix-examples"
---

### [Matrix examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#matrix-examples)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#matrix-examples "Permalink to this headline")

Generalizing, we define a matrix as any `Layout` that is rank-2. For example,

```console
Shape :  (4,2)
Stride:  (1,4)
  0   4
  1   5
  2   6
  3   7
```

is a 4x2 column-major layout with stride-1 down the columns and stride-4 across the rows, and

```console
Shape :  (4,2)
Stride:  (2,1)
  0   1
  2   3
  4   5
  6   7
```

is a 4x2 row-major layout with stride-2 down the columns and stride-1 across the rows. Majorness is simply which mode has stride-1.

Just like the vector layouts, each of the modes of the matrix can also be split into _multi-modes_.
This lets us express more layouts beyond just row-major and column-major. For example,

```console
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

is also logically 4x2, with stride-2 across the rows but a multi-stride down the columns. The first `2` elements down the column have a stride of `4` and then there is a copy of those with stride-1. Since this layout is logically 4x2,
like the column-major and row-major examples above,
we can _still_ use 2-D coordinates to index into it.
