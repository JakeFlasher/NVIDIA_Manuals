---
title: "4.7.4.3. Forcing a Module to Load Eagerly at Runtime"
section: "4.7.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#forcing-a-module-to-load-eagerly-at-runtime"
---

### [4.7.4.3. Forcing a Module to Load Eagerly at Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#forcing-a-module-to-load-eagerly-at-runtime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#forcing-a-module-to-load-eagerly-at-runtime "Permalink to this headline")

Loading kernels and variables happens automatically, without any need for explicit loading. Kernels can be loaded explicitly even without executing them by doing the following:

- The `cuModuleGetFunction()` function will cause a module to be loaded into device memory
- The `cudaFuncGetAttributes()` function will cause a kernel to be loaded into device memory

> **Note**
>
> `cuModuleLoad()` does not guarantee that a module will be loaded immediately.
