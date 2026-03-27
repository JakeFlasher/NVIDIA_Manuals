---
title: "3.5.2.2. Stream-Ordered Memory Allocation"
section: "3.5.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#stream-ordered-memory-allocation"
---

### [3.5.2.2. Stream-Ordered Memory Allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#stream-ordered-memory-allocation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#stream-ordered-memory-allocation "Permalink to this headline")

The [stream-ordered memory allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) allows programs to sequence allocation and freeing of GPU memory into a [CUDA stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams). Unlike `cudaMalloc` and `cudaFree` which execute immediately, `cudaMallocAsync` and `cudaFreeAsync` inserts a memory allocation  or free operation into a CUDA stream. [Section 4.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) covers all the details of these APIs.
