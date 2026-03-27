---
title: "Accumulator Mapping"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#0t_mma_atom--id4"
---

### [Accumulator Mapping](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#id4)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#id4 "Permalink to this headline")

Accumulators are mapped hierarchically in GMMA, starting from the concept of a core matrix and building up to a layout for the whole C matrix tile. Let’s look at this core matrix first. We only consider fp16 accumulators here, but extensions of fp32 accumulators as trivial as we will see later.

Each core matrix has the layout as shown in the diagram below.
![gmma_coremat_cd_fp16.png](images/___________--____1.png)

As in the Volta examples, the thread IDs are logical only, and which of the four warps they belong to in the warpgroup is not important.

Then GMMA tiles this core matrix first vertically along the M mode, and then repeats that column of core matrices along the N mode to construct the full MxN tile. This tiling is shown in the image below.

![gmma_wg_n_slice.png](images/___________--____2.png)

With this image, we are again ready to start building the `CLayout` for `SM90_64x128x16_F16F16F16F16_TN` atom. Same as before, we are constructing a mapping between the `(logical_thr_id, logical_val_id) -> (m, n)` coordinate spaces.

To begin, let’s follow the first few threads and values. We immediately see that they are arranged along the `N`-mode with pairs of values and four threads. This gives us

```cpp
// (T128,V4) -> (M64,N8)
using CLayout = Layout<Shape <Shape <  _4, ...>, Shape < _2, ...>>,
                       Stride<Stride<_128, ...>, Stride<_64, ...>>>;
```

To complete the first 8x8 core matrix, the four threads repeat eight times down the `M`-mode:

```cpp
// (T128,V4) -> (M64,N8)
using CLayout = Layout<Shape <Shape <  _4, _8, ...>, Shape < _2, ...>>,
                       Stride<Stride<_128, _1, ...>, Stride<_64, ...>>>;
```

Then, as we go to the next core matrix, we wrap back again to `T0`, but this time to `(T0, V2)`.

```cpp
// (T128,V4) -> (M64,N8)
using CLayout = Layout<Shape <Shape <  _4, _8, ...>, Shape < _2, _2>>,
                       Stride<Stride<_128, _1, ...>, Stride<_64, _8>>>;
```

Finally, we get this entire pattern repeating four times, once for each warp, down the `M`-mode starting at `(m,n) = (16,0) = 16`, where four core matrices that belong to the same warp are stacked on top of each other. This makes the size of the final sub-mode of `thrID` 4 (there are four warps) with a stride of `16` (to take us to coordinate `(16,0) = 16`).

```cpp
// (T128,V4) -> (M64,N8)
using CLayout = Layout<Shape <Shape <  _4, _8,  _4>, Shape < _2, _2>>,
                       Stride<Stride<_128, _1, _16>, Stride<_64, _8>>>;
```

This is the full `CLayout` for 64x8 accumulators. The GMMA instructions include 64xN variants with `N = [16,32,64,128,256]` where this 64x8 pattern is repeated giving each thread additional values. As this starts at `(m,n) = (0,8) = 512`, this is easy to account for in our `CLayout`. For example, the 64x128 `CLayout` is

```cpp
// (T128,V64) -> (M64,N128)
using CLayout = Layout<Shape <Shape <  _4, _8,  _4>, Shape < _2, _2,  _16>>,
                       Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;
```

where we see 16 copies of the 64x8 tile.
