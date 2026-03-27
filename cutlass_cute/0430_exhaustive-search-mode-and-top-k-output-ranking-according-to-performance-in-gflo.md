---
title: "Exhaustive search mode and top-k output ranking according to performance in GFLOPS/s"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflops-s"
---

## [Exhaustive search mode and top-k output ranking according to performance in GFLOPS/s](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflops-s)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflops-s "Permalink to this headline")

CUTLASS also allows a few options to enable searching best performing kernel in a broader parameter space.

1. **Sorting Performance Results by GFLOPs/second**
A new option enables users to sort the final performance report based on GFLOPs/second, making it easier to identify the most efficient kernels.
2. **Exhaustive Search for Best Kernel Performance in GFLOPs/second**
This feature allows the profiler to search for the best-performing kernel across a range of problem sizes, swizzle sizes, rasterization orders, and dynamic cluster configurations. It ensures that all viable configurations are considered to maximize performance.
3. **Performance Search Under a Fixed GEMM Shape**
This option enables exhaustive performance tuning for a specific problem size. Unlike the previous feature, this restricts the search to a fixed GEMM shape while still exploring various kernel parameters to find the best configuration.
