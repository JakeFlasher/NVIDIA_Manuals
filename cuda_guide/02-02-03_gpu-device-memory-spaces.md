---
title: "2.2.3. GPU Device Memory Spaces"
section: "2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#gpu-device-memory-spaces"
---

## [2.2.3. GPU Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#gpu-device-memory-spaces)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#gpu-device-memory-spaces "Permalink to this headline")

CUDA devices have several memory spaces that can be accessed by CUDA threads within kernels.  [Table 1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-memory-types-scopes-lifetimes) shows a summary of the common memory types, their thread scopes, and their lifetimes.  The following sections explain each of these memory types in more detail.

| Memory Type | Scope | Lifetime | Location |
| --- | --- | --- | --- |
| Global | Grid | Application | Device |
| Constant | Grid | Application | Device |
| Shared | Block | Kernel | SM |
| Local | Thread | Kernel | Device |
| Register | Thread | Kernel | SM |
