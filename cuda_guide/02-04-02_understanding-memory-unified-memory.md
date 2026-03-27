---
title: "2.4.2. Unified Memory"
section: "2.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#understanding-memory--unified-memory"
---

## [2.4.2. Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#unified-memory "Permalink to this headline")

_Unified memory_ is a CUDA memory feature which allows memory allocations called _managed memory_ to be accessed from code running on either the CPU or the GPU. Unified memory was shown in [the intro to CUDA in C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-unified-memory). Unified memory is available on all systems supported by CUDA.

On some systems, managed memory must be explicitly allocated. Managed memory can be explicitly allocated in CUDA in a few different ways:

- The CUDA API `cudaMallocManaged`
- The CUDA API `cudaMallocFromPoolAsync` with a pool created with `allocType` set to `cudaMemAllocationTypeManaged`
- Global variables with the `__managed__` specifier (see [Memory Space Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-space-specifiers))

On systems with [HMM](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management) or [ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services), all system memory is implicitly managed memory, regardless of how it is allocated. No special allocation is needed.
