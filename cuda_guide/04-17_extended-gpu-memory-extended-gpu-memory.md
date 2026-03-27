---
title: "4.17. Extended GPU Memory"
section: "4.17"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#extended-gpu-memory--extended-gpu-memory"
---

# [4.17. Extended GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#extended-gpu-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#extended-gpu-memory "Permalink to this headline")

The Extended GPU Memory (EGM) feature, utilizing the high-bandwidth
NVLink-C2C, facilitates efficient access to all system memory by GPUs,
in both single-node and multi-node systems.
EGM applies to integrated CPU-GPU NVIDIA systems by allowing physical memory
allocation that can be accessed from any GPU
thread within the setup. EGM ensures that all GPUs can access
its resources at the speed of either GPU-GPU NVLink or NVLink-C2C.

![EGM](images/________-___-______--________-___-_______1.png)

In this setup, memory accesses occur via the local high-bandwidth
NVLink-C2C. For remote memory accesses,
GPU NVLink and, in some cases, NVLink-C2C are used. With EGM, GPU
threads gain the capability to access all available memory resources,
including CPU attached memory and HBM3, over the NVSwitch fabric.
