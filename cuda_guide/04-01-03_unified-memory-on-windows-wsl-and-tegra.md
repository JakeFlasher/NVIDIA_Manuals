---
title: "4.1.3. Unified Memory on Windows, WSL, and Tegra"
section: "4.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-windows-wsl-and-tegra"
---

## [4.1.3. Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unified-memory-on-windows-wsl-and-tegra)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unified-memory-on-windows-wsl-and-tegra "Permalink to this headline")

> **Note**
>
> This section is only looking at devices with compute capability lower than 6.0 or Windows platforms, devices with  `concurrentManagedAccess` property set to 0.

Devices with compute capability lower than 6.0 or Windows platforms, devices with  `concurrentManagedAccess` property set to 0, see [Overview of Unified Memory Paradigms](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels),  support CUDA managed memory with the following limitations:

- **Data Migration and Coherency**: Fine-grained movement of the managed data to GPU on-demand is not supported. Whenever a GPU kernel is launched all managed memory generally has to be transferred to GPU memory to avoid faulting on memory access. Page faulting is only supported from the CPU side.
- **GPU Memory Oversubscription**: They cannot allocate more managed memory than the physical size of GPU memory.
- **Coherency and Concurrency**: Simultaneous access to managed memory is not possible, because coherence could not be guaranteed if the CPU accessed a unified memory allocation while a GPU kernel is active because of the missing GPU page faulting mechanism.
