---
title: "Types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#types"
---

### [Types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#types "Permalink to this headline")

The HMMA NT above uses types:

```cpp
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
```

The rest of the `MMA_Traits` will be described in units of these types.
