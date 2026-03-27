---
title: "5.2.2.6. CUDA_FORCE_PRELOAD_LIBRARIES"
section: "5.2.2.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-force-preload-libraries"
---

### [5.2.2.6. CUDA_FORCE_PRELOAD_LIBRARIES](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-force-preload-libraries)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-force-preload-libraries "Permalink to this headline")

The environment variable affects the preloading of libraries required for [NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) and [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation).

**Possible Values**:

- `1`: This forces the driver to preload the libraries required for [NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) and [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) during initialization. This increases the memory footprint and the time required for CUDA driver initialization. Setting this environment variable is necessary to avoid certain deadlock situations involving multiple threads.
- `0`: Default behavior.

**Example**:

```bash
CUDA_FORCE_PRELOAD_LIBRARIES=1
```
