---
title: "4.16. Virtual Memory Management"
section: "4.16"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management--virtual-memory-management"
---

# [4.16. Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#virtual-memory-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#virtual-memory-management "Permalink to this headline")

In the CUDA programming model, memory allocation calls (such as
`cudaMalloc()`) return a memory address that in GPU memory.
The address can be used with any CUDA API or inside a device
kernel.
Developers can enable peer device access to that memory
allocations by using  `cudaEnablePeerAccess`. By doing so,
kernels on different devices can access the same data. However,
all past and future user allocations are also mapped to the target peer device.
This can lead to users unintentionally paying a runtime cost for
mapping all `cudaMalloc` allocations to peer devices. In most
situations, applications communicate by sharing only a few allocations with
another device. It is usually not necessary to map all allocations to all
devices. In addition, extending this approach to multi-node settings becomes
inherently difficult.

CUDA provides a _virtual memory management_
(VMM) API to give developers explicit, low-level control over this process.

Virtual memory allocation, a complex process managed by the operating system
and the Memory Management Unit (MMU), works in two key stages. First, the OS
reserves a contiguous range of virtual addresses for a program without
assigning any physical memory. Then, when the program attempts to use that
memory for the first time, the OS commits the virtual addresses, assigning
physical storage to the virtual pages as needed.

CUDA’s VMM API brings a similar concept to GPU memory management by allowing
developers to explicitly reserve a virtual address range and then later map it
to physical GPU memory. With VMM, applications can specifically choose
certain allocations to be accessible by other devices.

The VMM API lets complex applications to manage
memory more efficiently across multiple GPUs (and CPU cores). By enabling
manual control over memory reservation, mapping, and access permissions, the
VMM API enables advanced techniques like fine-grained data sharing, zero-copy
transfers, and custom memory allocators. The CUDA VMM API expose fine grained
control to the user for managing the GPU memory in applications.

Developers can benefit from the VMM API in several key ways:

- Fine-grained control over virtual and physical memory management, allowing
allocation and mapping of non-contiguous physical memory chunks to
contiguous virtual address spaces. This helps reduce GPU memory
fragmentation and improve memory utilization, especially for large
workloads like deep neural network training.
- Efficient memory allocation and deallocation by separating the reservation
of virtual address space from the physical memory allocation. Developers
can reserve large virtual memory regions and map physical memory on demand
without costly memory copies or reallocations, leading to performance
improvements in dynamic data structures and variable-sized memory
allocations.
- The ability to grow GPU memory allocations dynamically without needing to
copy and reallocate all data, similar to how `realloc` or `std::vector` works
in CPU memory management. This supports more flexible and efficient GPU
memory use patterns.
- Enhancements to developer productivity and application performance by
providing low-level APIs that allow building sophisticated memory
allocators and cache management systems, such as dynamically managing
key-value caches in large language models, improving throughput and
latency.
- The CUDA VMM API is highly valuable in distributed multi-GPU settings as
it enables efficient memory sharing and access across multiple GPUs. By
decoupling virtual addresses from physical memory, the API allows
developers to create a unified virtual address space where data can be
dynamically mapped to different GPUs. This optimizes memory usage and
reduces data transfer overhead. For instance, NVIDIA’s libraries like NCCL,
and NVShmem actively uses VMM.

In summary, the CUDA VMM API gives developers advanced tools for fine-tuned,
efficient, flexible, and scalable GPU memory management beyond traditional
malloc-like abstractions, which is important for high-performance and
large-memory applications

> **Note**
>
> The suite of APIs described in this section require a system that
> supports UVA. See The [Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html).
