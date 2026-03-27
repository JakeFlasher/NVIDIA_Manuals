---
title: "4.7.1. Introduction"
section: "4.7.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading--introduction"
---

## [4.7.1. Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#introduction)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#introduction "Permalink to this headline")

Lazy loading reduces program initialization time by waiting to load CUDA modules until they are needed. Lazy loading is particularly effective for programs that only use a small number of the kernels they include, as is common when using libraries. Lazy loading is designed to be invisible to the user when the CUDA programming model is followed. [Potential Hazards](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#lazy-loading-potential-hazards) explains this in detail. As of CUDA 12.3 lazy Loading is enabled by default on all platforms, but can be controlled via the `CUDA_MODULE_LOADING` environment variable.
