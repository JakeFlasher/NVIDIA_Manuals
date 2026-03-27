---
title: "5.2.2.2. CUDA_CACHE_PATH"
section: "5.2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-path"
---

### [5.2.2.2. CUDA_CACHE_PATH](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-cache-path)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-cache-path "Permalink to this headline")

The environment variable specifies the directory path for the [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache.

**Possible Values**: The absolute path to the cache directory (with appropriate access permissions). The default values are:

- on Windows, `%APPDATA%\NVIDIA\ComputeCache`
- on Linux, `~/.nv/ComputeCache`

**Example**:

```bash
CUDA_CACHE_PATH=~/tmp
```
