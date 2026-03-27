---
title: "4.7.5.2. Large Memory Allocations"
section: "4.7.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#large-memory-allocations"
---

### [4.7.5.2. Large Memory Allocations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#large-memory-allocations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#large-memory-allocations "Permalink to this headline")

Lazy loading delays memory allocation for CUDA modules from program initialization until closer to execution time. If an application allocates the entire VRAM on startup, CUDA can fail to allocate memory for modules at runtime. Possible solutions:

- use `cudaMallocAsync()` instead of an allocator that allocates the entire VRAM on startup
- add some buffer to compensate for the delayed loading of kernels
- preload all kernels that will be used in the program before trying to initialize the allocator
