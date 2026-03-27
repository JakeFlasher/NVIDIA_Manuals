---
title: "2.4.3. Page-Locked Host Memory"
section: "2.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#page-locked-host-memory"
---

## [2.4.3. Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#page-locked-host-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#page-locked-host-memory "Permalink to this headline")

In [introductory code examples](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cuda-cpp-all-together), `cudaMallocHost` was used to allocate memory on the CPU. This allocates _page-locked_ memory (also known as _pinned_ memory) on the host. Host allocations made through traditional allocation mechanisms like `malloc`, `new`, or `mmap` are not page-locked, which means they may be swapped to disk or physically relocated by the operating system.

Page-locked host memory is required for [asynchronous copies between the CPU and GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#async-execution-memory-transfers). Page-locked host memory also improves performance of synchronous copies. Page-locked memory can be [mapped](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-mapped-memory) to the GPU for direct access from GPU kernels.

The CUDA runtime provides APIs to allocate page-locked host memory or to page-lock existing allocations:

- `cudaMallocHost` allocates page-locked host memory
- `cudaHostAlloc` defaults to the same behavior as `cudaMallocHost`, but also takes flags to specify other memory parameters
- `cudaFreeHost` frees memory allocated with `cudaMallocHost` or `cudaHostAlloc`
- `cudaHostRegister` page-locks a range of existing memory allocated outside the CUDA API, such as with `malloc` or `mmap`

`cudaHostRegister` enables host memory allocated by 3rd party libraries or other code outside of a developer’s control to be page-locked so that it can be used in asynchronous copies or mapped.

> **Note**
>
> Page-locked host memory can be used for asynchronous copies and mapped-memory by all GPUs in the system.
>
> Page-locked host memory is not cached on non I/O coherent Tegra devices. Also, `cudaHostRegister()` is not supported on non I/O coherent Tegra devices.
