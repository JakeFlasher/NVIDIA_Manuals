---
title: "4.7.5.1. Impact on Concurrent Kernel Execution"
section: "4.7.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-concurrent-kernel-execution"
---

### [4.7.5.1. Impact on Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#impact-on-concurrent-kernel-execution)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#impact-on-concurrent-kernel-execution "Permalink to this headline")

Some programs incorrectly assume that concurrent kernel execution is guaranteed. A deadlock can occur if cross-kernel synchronization is required, but kernel execution has been serialized. To minimize the impact of lazy loading on concurrent kernel execution, do the following:

- preload all kernels that you hope to execute concurrently prior to launching them or
- run application with `CUDA_MODULE_LOADING = EAGER` to force loading data eagerly without forcing each function to load eagerly
