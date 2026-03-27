---
title: "1.2.3.1. DRAM Memory in Heterogeneous Systems"
section: "1.2.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#dram-memory-in-heterogeneous-systems"
---

### [1.2.3.1. DRAM Memory in Heterogeneous Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#dram-memory-in-heterogeneous-systems)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#dram-memory-in-heterogeneous-systems "Permalink to this headline")

GPUs and CPUs both have directly attached DRAM chips. In systems with more than one GPU, each GPU has its own memory. From the perspective of device code, the DRAM attached to the GPU is called _global memory_, because it is accessible to all SMs in the GPU. This terminology does not mean it is necessarily accessible everywhere within the system. The DRAM attached to the CPU(s) is called _system memory_ or _host memory_.

Like CPUs, GPUs use virtual memory addressing. On all currently-supported systems, the CPU and GPU use a single unified virtual memory space. This means that the virtual memory address range for each GPU in the system is unique and distinct from the CPU and every other GPU in the system. For a given virtual memory address, it is possible to determine whether that address is in GPU memory or system memory and, on systems with multiple GPUs, which GPU memory contains that address.

There are CUDA APIs to allocate GPU memory, CPU memory, and to copy between allocations on the CPU and GPU, within a GPU, or between GPUs in multi-GPU systems. The locality of data can be explicitly controlled when desired. [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#programming-model-unified-memory), discussed below, allows the placement of memory to be handled automatically by the CUDA runtime or system hardware.
