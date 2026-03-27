---
title: "Layout Creation and Use"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layout-creation-and-use"
---

## [Layout Creation and Use](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#layout-creation-and-use)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#layout-creation-and-use "Permalink to this headline")

A `Layout` is a pair of `IntTuple`s: the `Shape` and the `Stride`. The first element defines the abstract _shape_ of the `Layout`, and the second element defines the _strides_, which map from coordinates within the shape to the index space.

We define many operations on `Layout`s analogous to those defined on `IntTuple`.

- `rank(Layout)`: The number of modes in a `Layout`. Equivalent to the tuple size of the `Layout`’s shape.
- `get<I>(Layout)`: The `I`th sub-layout of the `Layout`, with `I < rank`.
- `depth(Layout)`: The depth of the `Layout`’s shape. A single integer has depth 0, a tuple of integers has depth 1, a tuple of tuples of integers has depth 2, etc.
- `shape(Layout)`: The shape of the `Layout`.
- `stride(Layout)`: The stride of the `Layout`.
- `size(Layout)`: The size of the `Layout` function’s domain.  Equivalent to `size(shape(Layout))`.
- `cosize(Layout)`: The size of the `Layout` function’s codomain (not necessarily the range). Equivalent to `A(size(A) - 1) + 1`.
