---
title: "4.1.1.2.6. Overview of Memory Allocators for Unified Memory"
section: "4.1.1.2.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#overview-of-memory-allocators-for-unified-memory"
---

#### [4.1.1.2.6. Overview of Memory Allocators for Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#overview-of-memory-allocators-for-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#overview-of-memory-allocators-for-unified-memory "Permalink to this headline")

For systems with full CUDA unified memory support various different allocators may be used to allocate unified memory.
The following table shows an overview of a selection of allocators with their respective features. Note that all information in this section is subject to change in future CUDA versions.

| API | Placement Policy | Accessible From | Migrate Based On Access [^[2]] | Page Sizes [^[4]] [^[5]] |
| --- | --- | --- | --- | --- |
| `malloc`, `new`,  `mmap` | First touch/hint [^[1]] | CPU, GPU | Yes [^[3]] | System or huge page size [^[6]] |
| `cudaMallocManaged` | First touch/hint | CPU, GPU | Yes | CPU resident: system page size GPU resident: 2MB |
| `cudaMalloc` | GPU | GPU | No | GPU page size: 2MB |
| `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister` | CPU | CPU, GPU | No | Mapped by CPU: system page size   Mapped by GPU: 2MB |
| Memory pools, location type host: `cuMemCreate`, `cudaMemPoolCreate` | CPU | CPU, GPU | No | Mapped by CPU: system page size   Mapped by GPU: 2MB |
| Memory pools, location type device: `cuMemCreate`, `cudaMemPoolCreate`, `cudaMallocAsync` | GPU | GPU | No | 2MB |

[^[1]]: For `mmap`, file-backed memory is placed on the CPU by default, unless specified otherwise through `cudaMemAdviseSetPreferredLocation` (or `mbind`, see bullet points below).

[^[2]]: This feature can be overridden with `cudaMemAdvise`. Even if access-based migrations are disabled, if the backing memory space is full, memory might migrate.

[^[3]]: File-backed memory will not migrate based on access.

[^[4]]: The default system page size is 4KiB or 64KiB on most systems, unless huge page size was explicitly specified (for example, with `mmap` `MAP_HUGETLB` / `MAP_HUGE_SHIFT`). In this case, any huge page size configured on the system is supported.

[^[5]]: Page-sizes for GPU-resident memory may evolve in future CUDA versions.

[^[6]]: Currently huge page sizes may not be kept when migrating memory to the GPU or placing it through first-touch on the GPU.

The table [Overview of unified memory support of different allocators](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-um-allocators) shows the difference in semantics of several allocators that may be considered to allocate data accessible from multiple processors at a time, including host and device.
For additional details about `cudaMemPoolCreate`, see the [Memory Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-pools) section, for additional details about `cuMemCreate`, see the [Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management) section.

On hardware-coherent systems where device memory is exposed as a NUMA domain to the system, special allocators such as `numa_alloc_on_node` may be used to pin memory to the given NUMA node, either host or device. This memory is accessible from both host and device and does not migrate. Similarly,
`mbind` can be used to pin memory to the given NUMA node(s), and can cause file-backed memory to be placed on the given NUMA node(s) before it is first accessed.

The following applies to allocators of memory that is shared:

- System allocators such as `mmap` allow sharing the memory between processes using the `MAP_SHARED` flag. This is supported in CUDA and can be used to share memory between different devices connected to the same host. However, this is currently not supported for sharing memory between multiple hosts as well as multiple devices. See [Inter-Process Communication (IPC) with Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-fork-managed-memory) for details.
- For access to unified memory or other CUDA memory through a network on multiple hosts, consult the documentation of the communication library used, for example [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), [NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html), [OpenMPI](https://www.open-mpi.org/faq/?category=runcuda), [UCX](https://docs.mellanox.com/category/hpcx), etc.
