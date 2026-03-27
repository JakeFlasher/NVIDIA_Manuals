---
title: "Tiling a Tensor"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#tiling-a-tensor"
---

## [Tiling a Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tiling-a-tensor)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tiling-a-tensor "Permalink to this headline")

Many of the [`Layout` algebra operations](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) can also be applied to `Tensor`.

```cpp
   composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
 zipped_divide(Tensor, Tiler)
  tiled_divide(Tensor, Tiler)
   flat_divide(Tensor, Tiler)
```

The above operations allows arbitrary subtensors to be “factored out” of `Tensor`s. This very commonly used in tiling for threadgroups, tiling for MMAs, and reodering tiles of data for threads.

Note that the `_product` operations are not implemented for `Tensor`s as those would
often produce layouts with increased codomain sizes, which means the `Tensor` would
require accessing elements unpredictably far outside its previous bounds. `Layout`s can be
used in products, but not `Tensor`s.
