---
title: "4.3.2.1. Allocating Memory"
section: "4.3.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#allocating-memory"
---

### [4.3.2.1. Allocating Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#allocating-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#allocating-memory "Permalink to this headline")

The `cudaMallocAsync` function triggers asynchronous memory allocation on the
GPU, linked to a specific CUDA stream. `cudaMallocAsync` allows memory
allocation to occur without hindering the host or other
streams, eliminating the need for expensive synchronization.

> **Note**
>
> `cudaMallocAsync` ignores the current device/context when determining where
> the allocation will reside. Instead, `cudaMallocAsync` determines the
> appropriate device based on the specified memory pool or the supplied stream.

The listing below illustrates a fundamental use pattern: the memory is
allocated, used, and then freed back into the same stream.

```c++
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// do work using the allocation
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// An asynchronous free can be specified without synchronizing the cpu and GPU
cudaFreeAsync(ptr, cudaStreamPerThread);
```

> **Note**
>
> When accessing allocation from a stream other than the stream that made the allocation, the
> user must guarantee that the access occurs after the allocation
> operation, otherwise the behavior is undefined.
