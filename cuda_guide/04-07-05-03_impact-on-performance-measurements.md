---
title: "4.7.5.3. Impact on Performance Measurements"
section: "4.7.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-performance-measurements"
---

### [4.7.5.3. Impact on Performance Measurements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#impact-on-performance-measurements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#impact-on-performance-measurements "Permalink to this headline")

Lazy loading may skew performance measurements by moving CUDA module initialization into the measured execution window. To avoid this:

- do at least one warmup iteration prior to measurement
- preload the benchmarked kernel prior to launching it
