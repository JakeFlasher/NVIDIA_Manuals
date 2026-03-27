---
title: "Zipped and Tiled Products"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#zipped-and-tiled-products"
---

### [Zipped and Tiled Products](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#zipped-and-tiled-products)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#zipped-and-tiled-products "Permalink to this headline")

Similar to `zipped_divide` and `tiled_divide`, the `zipped_product` and `tiled_product` simply rearrange the modes that result from a by-mode `logical_product`.

```text
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
```
