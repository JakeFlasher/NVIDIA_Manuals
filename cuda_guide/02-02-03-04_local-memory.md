---
title: "2.2.3.4. Local Memory"
section: "2.2.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#local-memory"
---

### [2.2.3.4. Local Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#local-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#local-memory "Permalink to this headline")

Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space.  The ‘local’ label refers to its logical scope, not its physical location.  Local memory is used for thread local storage during the execution of a kernel.  Automatic variables that the compiler is likely to place in local memory are:

- Arrays for which it cannot determine that they are indexed with constant quantities,
- Large structures or arrays that would consume too much register space,
- Any variable if the kernel uses more registers than available, that is register spilling.

Because the local memory space resides in device memory, local memory accesses have the same latency and bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in [Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-coalesced-global-memory-access). Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address, such as the same index in an array variable or the same member in a structure variable.
