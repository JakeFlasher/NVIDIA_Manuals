---
title: "Examples"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#examples"
---

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#examples)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#examples "Permalink to this headline")

```mlir
!cuda_tile.tensor_view<1024x1024xf16, strides=[1024,1]>
!cuda_tile.tensor_view<32x16x32xf16, strides=[512,1,16]>
!cuda_tile.tensor_view<?x?xf16, strides=[?,1]>
!cuda_tile.tensor_view<?x16xf32, strides=[1,?]>
```
