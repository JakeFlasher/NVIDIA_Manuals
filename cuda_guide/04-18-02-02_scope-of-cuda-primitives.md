---
title: "4.18.2.2. Scope of CUDA Primitives"
section: "4.18.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#scope-of-cuda-primitives"
---

### [4.18.2.2. Scope of CUDA Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#scope-of-cuda-primitives)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#scope-of-cuda-primitives "Permalink to this headline")

CUDA Dynamic Parallelism relies on the [CUDA Device Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cuda-device-runtime), which enables calling a limited set of APIs which are syntactically similar to the CUDA Runtime API, but available in device code. The behavior of the device runtime APIs are similar to their host counterparts, but there are some differences. These differences are captured in the table in section [API Reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-runtime-api-reference).

On both host and device, the CUDA runtime offers an API for launching kernels and for tracking dependencies between launches via streams and events. On the device, launched kernels and CUDA objects are visible to all threads in the invoking grid. This means, for example, that a stream may be created by one thread and used by any other thread in the same grid. However, CUDA objects such as streams and events which were created on by device API calls are only valid within the grid where they were created.
