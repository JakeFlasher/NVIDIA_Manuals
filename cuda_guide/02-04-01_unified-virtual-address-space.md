---
title: "2.4.1. Unified Virtual Address Space"
section: "2.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-virtual-address-space"
---

## [2.4.1. Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#unified-virtual-address-space)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#unified-virtual-address-space "Permalink to this headline")

A single virtual address space is used for all host memory and all global memory on all GPUs in the system within a single OS process. All memory allocations on the host and on all devices lie in this virtual address space. This is true whether allocations are made with CUDA APIs (e.g. `cudaMalloc`, `cudaMallocHost`) or with system allocation APIs (e.g. `new`, `malloc`, `mmap`). The CPU and each GPU has a unique range within the unified virtual address space.

This means:

- The location of any memory (that is, CPU or which GPU’s memory it lies in) can be determined from the value of a pointer using `cudaPointerGetAttributes()`
- The `cudaMemcpyKind` parameter of `cudaMemcpy*()` can be set to `cudaMemcpyDefault` to automatically determine the copy type from the pointers
