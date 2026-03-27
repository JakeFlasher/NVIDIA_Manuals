---
title: "3.5.5.1. Virtual Memory Management"
section: "3.5.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#virtual-memory-management"
---

### [3.5.5.1. Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#virtual-memory-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#virtual-memory-management "Permalink to this headline")

As mentioned in [Section 2.4.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-virtual-address-space), all GPUs in a system, along with the CPU memory, share a single unified virtual address space. Most applications can use the default memory management provided by CUDA without the need to change its behavior. However,  [the CUDA driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api) provides advanced and detailed controls over the layout of this virtual memory space for those that need it. This is mostly applicable for controlling the behavior of buffers when sharing between GPUs both within and across multiple systems.

[Section 4.16](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management) covers the controls offered by the CUDA driver API, how they work and when a developer may find them advantageous.
