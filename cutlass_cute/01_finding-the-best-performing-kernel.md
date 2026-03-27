---
title: "1. Finding the Best Performing Kernel"
section: "1"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#finding-the-best-performing-kernel"
---

#### [1. Finding the Best Performing Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#finding-the-best-performing-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#finding-the-best-performing-kernel "Permalink to this headline")

Use the following command to conduct an exhaustive search and sort results by GFLOPs/second:

```bash
cutlass_profiler --kernels=*gemm* --enable-kernel-performance-search --sort-results-flops-per-sec
```
