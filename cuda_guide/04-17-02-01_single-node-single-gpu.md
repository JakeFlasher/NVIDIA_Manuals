---
title: "4.17.2.1. Single-Node, Single-GPU"
section: "4.17.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#single-node-single-gpu"
---

### [4.17.2.1. Single-Node, Single-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#single-node-single-gpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#single-node-single-gpu "Permalink to this headline")

Any of the existing CUDA host allocators as well as system allocated
memory can be used to benefit from high-bandwidth C2C. To the user,
local access is what a host allocation is today.

> **Note**
>
> Refer to the tuning guide for more information about memory allocators and page sizes.
