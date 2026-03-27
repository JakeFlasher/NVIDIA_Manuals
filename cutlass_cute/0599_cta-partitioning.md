---
title: "CTA Partitioning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning"
---

### [CTA Partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cta-partitioning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cta-partitioning "Permalink to this headline")

Now that we have the representations of the full matrices, it’s time to tile them and split up the work!

At the highest level, the work is distributed across CTAs. In principle, each CTA’s tile could come from the input tensors in many different ways. Many [CuTe `Tiler`s](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition-tilers) could be used to tile the data, but for these cases it is sufficient to simply use the shape of the desired CTA tile.

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)
```

Once the tiler has been defined, we can use it to tile and partition the tensors across the CTAs.

```cpp
  // Get the appropriate blocks for this threadblock
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

First, the CTA coordinate is created.

- The `m`-coordinate of this tile is given by `blockIdx.x`.
- The `n`-coordinate of this tile is given by `blockIdx.y`.
- The `k`-coordinate of this tile is unspecified – we want all of the tiles in `K` so the coordinate is `_`, the `Underscore` value, to keep that mode.

Then, `local_tile` is used to remove the modes of the tiler and coord corresponding to the `X`s. That is, the `Step<_1, X,_1>` is just shorthand for

```cpp
  // Use select<0,2> to use only the M- and K-modes of the tiler and coord
  Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
```

This `local_tile` is simply shorthand for

1. apply the tiler via [`zipped_divide`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#zipped-tiled-flat-divides)

```cpp
// ((BLK_M,BLK_K),(m,k))
Tensor gA_mk = zipped_divide(mA, select<0,2>(cta_tiler));
```

2. apply the coord to the second mode, the “Rest” mode, to extract out the correct tiles for this CTA.

```cpp
// (BLK_M,BLK_K,k)
Tensor gA = gA_mk(make_coord(_,_), select<0,2>(cta_coord));
```

Because the projections of the tiler and coord are symmetric and the two steps (apply a tiler and then slice into the rest-mode to produce a partition) are so common, they are wrapped together into the projective `local_tile` interface.

For tensor `A`, we are left with a rank-3 tensor of shape `(BLK_M,BLK_K,k)`. The first two modes are precisely the modes of the CTA tile and the last mode indexes over all of the tiles that will be reduced by this CTA. In the mainloop section below, this mode is iterated over via the `k_tile` loop.
