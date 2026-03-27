---
title: "4.13.7. Query L2 cache Properties"
section: "4.13.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#query-l2-cache-properties"
---

## [4.13.7. Query L2 cache Properties](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#query-l2-cache-properties)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#query-l2-cache-properties "Permalink to this headline")

Properties related to L2 cache are a part of `cudaDeviceProp` struct and can be queried using CUDA runtime API `cudaGetDeviceProperties`

CUDA Device Properties include:

- `l2CacheSize`: The amount of available L2 cache on the GPU.
- `persistingL2CacheMaxSize`: The maximum amount of L2 cache that can be set-aside for persisting memory accesses.
- `accessPolicyMaxWindowSize`: The maximum size of the access policy window.
