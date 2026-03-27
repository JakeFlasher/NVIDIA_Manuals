---
title: "2.2. Writing CUDA SIMT Kernels"
section: "2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-simt-kernels"
---

# [2.2. Writing CUDA SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#writing-cuda-simt-kernels)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-simt-kernels "Permalink to this headline")

CUDA C++ kernels can largely be written in the same way that traditional CPU code would be written for a given problem. However, there are some unique features of the GPU that can be used to improve performance. Additionally, some understanding of how threads on the GPU are scheduled, how they access memory, and how their execution proceeds can help developers write kernels that maximize utilization of the available computing resources.
