---
title: "Shape"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#shape"
---

### [Shape](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#shape)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#shape "Permalink to this headline")

The HMMA NT above has shape 8x8x4:

```cpp
  // Logical shape of the MMA
  using Shape_MNK = Shape <_8,_8,_4>;
```
