---
title: "4.3.2. Memory Management"
section: "4.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#memory-management"
---

## [4.3.2. Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-management "Permalink to this headline")

`cudaMallocAsync` and `cudaFreeAsync` are the APIs which enable stream-ordered memory management.
`cudaMallocAsync` returns an allocation and `cudaFreeAsync` frees an
allocation. Both APIs accept stream arguments to define when the allocation
will become and stop being available for use.
These functions allow memory operations to be tied to specific CUDA streams,
enabling them to occur without blocking the host or other
streams.  Application performance can be improved by avoiding potentially
costly synchronization of `cudaMalloc` and `cudaFree`.

These APIs can be used for further performance optimization through
memory pools, which manage and reuse large blocks of memory for more efficient
allocation and deallocation. Memory pools help reduce overhead and prevent
fragmentation, improving performance in scenarios with frequent memory
allocation operations.
