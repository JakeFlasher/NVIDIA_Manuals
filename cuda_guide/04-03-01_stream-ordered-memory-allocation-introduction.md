---
title: "4.3.1. Introduction"
section: "4.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocation--introduction"
---

## [4.3.1. Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#introduction)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#introduction "Permalink to this headline")

Managing memory allocations using `cudaMalloc` and `cudaFree` causes the GPU
to synchronize across all executing CUDA streams. The stream-ordered memory
allocator enables applications to order memory allocation and deallocation
with other work launched into a CUDA stream such as kernel launches and
asynchronous copies. This improves application memory use by taking advantage
of stream-ordering semantics to reuse memory allocations. The allocator also
allows applications to control the allocator’s memory caching behavior. When
set up with an appropriate release threshold, the caching behavior allows the
allocator to avoid expensive calls into the OS when the application indicates
it is willing to accept a bigger memory footprint. The allocator also supports
easy and secure allocation sharing between processes.

Stream-Ordered Memory Allocator:

> - Reduces the need for custom memory management abstractions, and makes it
> easier to create high-performance custom memory management for
> applications that need it.
> - Enables multiple libraries to share a common memory pool managed by the
> driver. This can reduce excess memory consumption.
> - Allows, the driver to perform optimizations based on its awareness of the
> allocator and other stream management APIs.

> **Note**
>
> Nsight Compute and the Next-Gen CUDA debugger is aware of the allocator
> since CUDA 11.3.
