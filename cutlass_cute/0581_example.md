---
title: "Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#example"
---

#### [Example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example "Permalink to this headline")

The specialization of MMA_Traits for the
`SM70_8x8x4_F32F16F16F32_NT` Operation lives in the header file
[`include/cute/atom/mma_traits_sm70.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/mma_traits_sm70.hpp).
It looks like this.

```c++
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};
```

The next section will explain these type aliases in detail.
