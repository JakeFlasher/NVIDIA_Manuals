---
title: "Threadblock Rasterization"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#threadblock-rasterization"
---

### [Threadblock Rasterization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#threadblock-rasterization)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#threadblock-rasterization "Permalink to this headline")

To maximize reuse of data held in the last level cache, CUTLASS defines several functions to
affect the mapping of threadblocks to logical partitions of the GEMM problem. These map
consecutively launched threadblocks to packed two-dimensional regions of the partitioned GEMM
problem to increase the probability that these will access the same tiles of global memory at
approximately the same time.

Several functions are defined in [cutlass/gemm/threadblock_swizzle.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h).
