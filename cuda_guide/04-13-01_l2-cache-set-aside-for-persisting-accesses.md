---
title: "4.13.1. L2 Cache Set-Aside for Persisting Accesses"
section: "4.13.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#l2-cache-set-aside-for-persisting-accesses"
---

## [4.13.1. L2 Cache Set-Aside for Persisting Accesses](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#l2-cache-set-aside-for-persisting-accesses)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#l2-cache-set-aside-for-persisting-accesses "Permalink to this headline")

A portion of the L2 cache can be set aside to be used for persisting data accesses to global memory. Persisting accesses have prioritized use of this set-aside portion of L2 cache, whereas normal or streaming accesses to global memory can only utilize this portion of L2 when it is unused by persisting accesses.

The L2 cache set-aside size for persisting accesses may be adjusted, within limits:

```c++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```

When the GPU is configured in Multi-Instance GPU (MIG) mode, the L2 cache set-aside functionality is disabled.

When using the Multi-Process Service (MPS), the L2 cache set-aside size cannot be changed by `cudaDeviceSetLimit`. Instead, the set-aside size can only be specified at start up of MPS server through the environment variable `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`.
