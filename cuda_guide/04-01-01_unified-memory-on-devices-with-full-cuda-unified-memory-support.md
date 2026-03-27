---
title: "4.1.1. Unified Memory on Devices with Full CUDA Unified Memory Support"
section: "4.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-devices-with-full-cuda-unified-memory-support"
---

## [4.1.1. Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unified-memory-on-devices-with-full-cuda-unified-memory-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unified-memory-on-devices-with-full-cuda-unified-memory-support "Permalink to this headline")

These systems include hardware-coherent memory systems, such as NVIDIA Grace Hopper and modern Linux systems with Heterogeneous Memory Management (HMM) enabled.
HMM is a software-based memory management system, providing the same programming model as hardware-coherent memory systems.

Linux HMM requires Linux kernel version 6.1.24+, 6.2.11+ or 6.3+, devices with compute capability 7.5 or higher and a CUDA driver version 535+ installed with [Open Kernel Modules](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#nvidia-open-gpu-kernel-modules).

> **Note**
>
> We refer to systems with a combined page table for both CPUs and GPUs as _hardware
> coherent_ systems. Systems with separate page tables for CPUs and GPUs are
> referred to as _software-coherent_.

Hardware-coherent systems such as NVIDIA Grace Hopper offer a logically combined page table for both CPUs and GPUs, see [CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-hw-coherency).
The following section only applies to hardware-coherent systems:

> - [Access Counter Migration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-access-counters)
