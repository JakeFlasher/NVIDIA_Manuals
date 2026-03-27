---
title: "2.4.3.1.3. Comparing Unified Memory and Mapped Memory"
section: "2.4.3.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#comparing-unified-memory-and-mapped-memory"
---

#### [2.4.3.1.3. Comparing Unified Memory and Mapped Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#comparing-unified-memory-and-mapped-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#comparing-unified-memory-and-mapped-memory "Permalink to this headline")

Mapped memory makes CPU memory accessible from the GPU, but does not guarantee that all types of access, for example atomics, are supported on all systems. Unified memory guarantees that all access types are supported.

Mapped memory remains in CPU memory, which means all GPU accesses must go through the connection between the CPU and GPU: PCIe or NVLink. Latency of accesses made across these links are significantly higher than access to GPU memory, and total available bandwidth is lower. As such, using mapped memory for all kernel memory accesses is unlikely to fully utilize GPU computing resources.

Unified memory is most often migrated to the physical memory of the processor accessing it. After the first migration, repeated access to the same memory page or cache line by a kernel can utilize the full GPU memory bandwidth.

> **Note**
>
> Mapped memory has also been referred to as _zero-copy_ memory in previous documents.
>
> Prior to all CUDA applications using a [unified virtual address space](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-virtual-address-space), additional APIs were needed to enable memory mapping (`cudaSetDeviceFlags` with `cudaDeviceMapHost`).  These APIs are no longer needed.
>
> Atomic functions (see [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions)) operating on mapped host memory are not atomic from the point of view of the host or other GPUs.
>
> CUDA runtime requires that 1-byte, 2-byte, 4-byte, 8-byte, and 16-byte naturally aligned loads and stores to host memory initiated from the device are preserved as single accesses from the point of view of the host and other devices. On some platforms, atomics to memory may be broken by the hardware into separate load and store operations. These component load and store operations have the same requirements on preservation of naturally aligned accesses. The CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits 8-byte naturally aligned operations and NVIDIA is not aware of any topology that splits 16-byte naturally aligned operations.
