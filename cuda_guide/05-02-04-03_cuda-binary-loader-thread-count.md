---
title: "5.2.4.3. CUDA_BINARY_LOADER_THREAD_COUNT"
section: "5.2.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-binary-loader-thread-count"
---

### [5.2.4.3. CUDA_BINARY_LOADER_THREAD_COUNT](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-binary-loader-thread-count)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-binary-loader-thread-count "Permalink to this headline")

Sets the number of CPU threads to use when loading device binaries. When set to 0, the number of CPU threads used is set to a default value of 1.

**Possible Values**:

> - Integer number of threads to use. Defaults to 0, which uses 1 thread.

**Example**:

```bash
CUDA_BINARY_LOADER_THREAD_COUNT=4
```
