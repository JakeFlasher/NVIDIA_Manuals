---
title: "5.2.2.3. CUDA_CACHE_MAXSIZE"
section: "5.2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-maxsize"
---

### [5.2.2.3. CUDA_CACHE_MAXSIZE](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-cache-maxsize)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-cache-maxsize "Permalink to this headline")

The environment variable specifies the [Just-In-Time (JIT) compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache size in bytes. Binaries that exceed this size are not cached. If needed, older binaries are evicted from the cache to make room for newer ones.

**Possible Values**: Number of bytes. The default values are:

- On desktop/server platforms, `1073741824` (1 GiB)
- On embedded platforms, `268435456` (256 MiB)

`4294967296` (4 GiB) is the maximum size.

**Example**:

```bash
CUDA_CACHE_MAXSIZE=268435456 # 256 MiB
```
