---
title: "2.4.2.2.1. Full Unified Memory with Hardware Coherency"
section: "2.4.2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#full-unified-memory-with-hardware-coherency"
---

#### [2.4.2.2.1. Full Unified Memory with Hardware Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#full-unified-memory-with-hardware-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#full-unified-memory-with-hardware-coherency "Permalink to this headline")

On hardware such as Grace Hopper and Grace Blackwell, where an NVIDIA CPU is used and the interconnect between the CPU and GPU is NVLink Chip-to-Chip (C2C), address translation services (ATS) are available. `cudaDevAttrPageableMemoryAccessUsesHostPageTables` is 1 when ATS is available.

With ATS, in addition to full unified memory support for all host allocations:

- GPU allocations (e.g. `cudaMalloc`) can be accessed from the CPU (`cudaDevAttrDirectManagedMemAccessFromHost` will be 1)
- The link between CPU and GPU supports native atomics (`cudaDevAttrHostNativeAtomicSupported` will be 1)
- Hardware support for coherence can improve performance compared to software coherence

ATS provides all capabilities of [HMM](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management). When ATS is available, HMM is automatically disabled. Further discussion of hardware vs. software coherency is found in [CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-hw-coherency).
