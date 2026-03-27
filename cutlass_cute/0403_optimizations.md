---
title: "Optimizations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#optimizations"
---

## [Optimizations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#optimizations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#optimizations "Permalink to this headline")

The hierarchical structure described above yields an efficient mapping to the CUDA execution model and
CUDA/TensorCores in NVIDIA GPUs. The following sections describe strategies for obtaining peak performance
for all corners of the design space, maximizing parallelism and exploiting data locality wherever possible.
