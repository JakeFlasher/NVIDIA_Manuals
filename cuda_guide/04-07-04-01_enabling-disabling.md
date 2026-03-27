---
title: "4.7.4.1. Enabling & Disabling"
section: "4.7.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#enabling-disabling"
---

### [4.7.4.1. Enabling & Disabling](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#enabling-disabling)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#enabling-disabling "Permalink to this headline")

Lazy loading is enabled by setting the `CUDA_MODULE_LOADING` environment variable to `LAZY`. Lazy loading can be disabled by setting the `CUDA_MODULE_LOADING` environment variable to `EAGER`. As of CUDA 12.3, lazy loading is enabled by default on all platforms.
