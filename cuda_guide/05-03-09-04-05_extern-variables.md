---
title: "5.3.9.4.5. extern Variables"
section: "5.3.9.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extern-variables"
---

#### [5.3.9.4.5. extern Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#extern-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#extern-variables "Permalink to this headline")

When compiling in the [whole program compilation mode](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation), `__device__`, `__shared__`, `__managed__`, and `__constant__` variables cannot be defined with external linkage using the `extern` keyword.

The only exception is for dynamically allocated `__shared__` variables as described in the [Dynamic Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-dynamic-allocation-shared-memory) section.

```cuda
__device__        int x; // OK
extern __device__ int y; // ERROR in whole program compilation mode
extern __shared__ int z; // OK
```
