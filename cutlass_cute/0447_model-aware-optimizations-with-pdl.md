---
title: "Model-Aware Optimizations with PDL"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html#model-aware-optimizations-with-pdl"
---

## [Model-Aware Optimizations with PDL](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#model-aware-optimizations-with-pdl)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#model-aware-optimizations-with-pdl "Permalink to this headline")

In [example 63](https://github.com/NVIDIA/cutlass/tree/main/examples/63_hopper_gemm_with_weight_prefetch/README.md), we use PDL to explicitly optimize for
performance of kernels where we know that one of the input matrices (our weights) will not be produced by a prior
kernel. In that case, we only need to wait on the prior kernels memory flush in order to load the other input matrix
(our activations). During our prologue, we can prefetch our weights to improve performance for memory bandwidth-bound
problem sizes. For more information, we refer the reader to [the example](https://github.com/NVIDIA/cutlass/tree/main/examples/63_hopper_gemm_with_weight_prefetch/README.md).
