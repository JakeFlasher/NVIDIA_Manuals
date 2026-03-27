---
title: "2.4. Unified and System Memory"
section: "2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-and-system-memory"
---

# [2.4. Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#unified-and-system-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#unified-and-system-memory "Permalink to this headline")

Heterogeneous systems have multiple physical memories where data can be stored. The host CPU has attached DRAM, and every GPU in a system has its own attached DRAM. Performance is best when data is resident in the memory of the processor accessing it. CUDA provides APIs to [explicitly manage memory placement](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-explicit-memory-management), but this can be verbose and complicate software design. CUDA provides features and capabilities aimed at easing allocation, placement, and migration of data between different physical memories.

The purpose of this chapter is to introduce and explain these features and what they mean to application developers for both functionality and performance. Unified memory has several different manifestations which depend upon the OS, driver version, and GPU used. This chapter will show how to determine which unified memory paradigm applies and how the features of unified memory behave in each. The later [chapter on unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro) explains unified memory in more detail.

The following concepts will be defined and explained in this chapter:

- [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-virtual-address-space) - CPU memory and each GPU’s memory have a distinct range within a single virtual address space
- [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory) - A CUDA feature that enables managed memory which can be automatically migrated between CPU and GPUs

> - [Limited Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-limited-unified-memory-support) - A unified memory paradigm with some limitations
> - [Full Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory-full) - Full support for unified memory features
> - [Full Unified Memory with Hardware Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services) - Full support for unified memory using hardware capabilities
> - [Unified memory hints](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-mem-advise-prefetch) - APIs to guide unified memory behavior for specific allocations
- [Page-locked Host Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-page-locked-host-memory) - Non-pageable system memory, which is necessary for some CUDA operations

> - [Mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-mapped-memory) - A mechanism (different from unified memory) for accessing host memory directly from a kernel

Additionally, the following terms used when discussing unified and system memory are introduced here:

- [Heterogeneous Managed Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management) (HMM) - A feature of the Linux kernel that enables software coherency for full unified memory
- [Address Translation Services](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services) (ATS) - A hardware feature, available when GPUs are connected to the CPU by the NVLink Chip-to-Chip (C2C) interconnect, which provides hardware coherency for full unified memory
