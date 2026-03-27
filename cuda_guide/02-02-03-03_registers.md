---
title: "2.2.3.3. Registers"
section: "2.2.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#registers"
---

### [2.2.3.3. Registers](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#registers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#registers "Permalink to this headline")

Registers are located on the SM and have thread local scope.  Register usage is managed by the compiler and registers are used for thread local storage during the execution of a kernel.  The number of registers per SM and the number of registers per thread block can be queried using the `regsPerMultiprocessor` and `regsPerBlock` device properties of the GPU.

NVCC allows the developer to [specify a maximum number of registers](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#maxrregcount-amount-maxrregcount) to be used by a kernel via the `-maxrregcount` option.  Using this option to reduce the number of registers a kernel can use may result in more thread blocks being scheduled on the SM concurrently, but may also result in more register spilling.
