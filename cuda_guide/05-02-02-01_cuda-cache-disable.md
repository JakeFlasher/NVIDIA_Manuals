---
title: "5.2.2.1. CUDA_CACHE_DISABLE"
section: "5.2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-disable"
---

### [5.2.2.1. CUDA_CACHE_DISABLE](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-cache-disable)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-cache-disable "Permalink to this headline")

The environment variable controls the behavior of the on-disk [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache.
Disabling the JIT cache forces PTX to CUBIN compilation for a CUDA application each time it is executed, unless the CUBIN code for the running architecture is found in the binary.

Disabling the JIT cache increases an application’s load time during initial execution. However, it can be useful for reducing the application’s disk space and for diagnosing differences across driver versions or build flags.

**Possible Values**:

- `1`: Disables PTX JIT caching.
- `0`: Enables PTX JIT caching (default).

**Examples**:

```bash
CUDA_CACHE_DISABLE=1 # disables caching
CUDA_CACHE_DISABLE=0 # enables caching
```
