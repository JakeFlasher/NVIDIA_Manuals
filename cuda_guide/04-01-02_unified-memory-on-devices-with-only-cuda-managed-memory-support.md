---
title: "4.1.2. Unified Memory on Devices with only CUDA Managed Memory Support"
section: "4.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-devices-with-only-cuda-managed-memory-support"
---

## [4.1.2. Unified Memory on Devices with only CUDA Managed Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unified-memory-on-devices-with-only-cuda-managed-memory-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unified-memory-on-devices-with-only-cuda-managed-memory-support "Permalink to this headline")

For devices with compute capability 6.x or higher but without pageable memory access,  see table [Overview of Unified Memory Paradigms](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels), CUDA managed memory is fully supported and coherent but the GPU cannot access system-allocated memory. The programming model and performance tuning of unified memory is largely similar to the model as described in the section, [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-pageable-systems),
with the notable exception that system allocators cannot be used to allocate memory. Thus, the following list of sub-sections do not apply:

- [Unified Memory: In-Depth Examples](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-system-allocator)
- [CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-hw-coherency)
- [Atomic Accesses and Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-atomics)
- [Access Counter Migration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-access-counters)
- [Avoid Frequent Writes to GPU-Resident Memory from the CPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-traffic-hd)
- [Exploiting Asynchronous Access to System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-async-access)
