---
title: "2.4.2.4. Memory Advise and Prefetch"
section: "2.4.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-advise-and-prefetch"
---

### [2.4.2.4. Memory Advise and Prefetch](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#memory-advise-and-prefetch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-advise-and-prefetch "Permalink to this headline")

The programmer can provide hints to the NVIDIA Driver managing unified memory to help it maximize application performance. The CUDA API `cudaMemAdvise` allows the programmer to specify properties of allocations that affect where they are placed and whether or not the memory is migrated when accessed from another device.

`cudaMemPrefetchAsync` allows the programmer to suggest an asynchronous migration of a specific allocation to a different location be started. A common use is starting the transfer of data a kernel will use before the kernel is launched. This enables the copy of data to occur while other GPU kernels are executing.

The section on [Performance Hints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-perf-hints) covers the different hints that can be passed to `cudaMemAdvise` and shows examples of using `cudaMemPrefetchAsync`.
