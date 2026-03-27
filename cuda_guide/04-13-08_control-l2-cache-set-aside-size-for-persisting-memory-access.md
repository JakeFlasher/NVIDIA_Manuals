---
title: "4.13.8. Control L2 Cache Set-Aside Size for Persisting Memory Access"
section: "4.13.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#control-l2-cache-set-aside-size-for-persisting-memory-access"
---

## [4.13.8. Control L2 Cache Set-Aside Size for Persisting Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#control-l2-cache-set-aside-size-for-persisting-memory-access)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#control-l2-cache-set-aside-size-for-persisting-memory-access "Permalink to this headline")

The L2 set-aside cache size for persisting memory accesses is queried using CUDA runtime API `cudaDeviceGetLimit` and set using CUDA runtime API `cudaDeviceSetLimit` as a `cudaLimit`. The maximum value for setting this limit is `cudaDeviceProp::persistingL2CacheMaxSize`.

```c++
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```
