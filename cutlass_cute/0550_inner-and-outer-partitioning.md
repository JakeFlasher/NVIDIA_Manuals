---
title: "Inner and outer partitioning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#inner-and-outer-partitioning"
---

### [Inner and outer partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#inner-and-outer-partitioning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#inner-and-outer-partitioning "Permalink to this headline")

Let’s take a tiled example and look at how we can slice it in useful ways.

```cpp
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```

Suppose that we want to give each threadgroup one of these 4x8 tiles of data. Then we can use our threadgroup coordinate to index into the second mode.

```cpp
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

We call this an _inner-partition_ because it keeps the inner “tile” mode. This pattern of applying a tiler and then slicing out that tile by indexing into the remainder mode is common and has been wrapped into its own function `inner_partition(Tensor, Tiler, Coord)`. You’ll often see `local_tile(Tensor, Tiler, Coord)` which is just another name for `inner_partition`. The `local_tile` partitioner is very often applied at the threadgroup level to partition tensors into tiles across threadgroups.

Alternatively, suppose that we have 32 threads and want to give each thread one element of these 4x8 tiles of data. Then we can use our thread to index into the first mode.

```cpp
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

We call this an _outer-partition_ because it keeps the outer “rest” mode. This pattern of applying a tiler and then slicing into that tile by indexing into the tile mode is common and has been wrapped into its own function `outer_partition(Tensor, Tiler, Coord)`. Sometimes you’ll see `local_partition(Tensor, Layout, Idx)`, which is a rank-sensitive wrapper around `outer_partition` that transforms the `Idx` into a `Coord` using the inverse of the `Layout` and then constructs a `Tiler` with the same top-level shape of the `Layout`. This allows the user to ask for a row-major, column-major, or arbitrary layout of threads with a given shape that can be used to partition into a tensor.

To see how these partitioning patterns are used, see the [introductory GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html).
