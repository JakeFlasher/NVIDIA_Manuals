---
title: "2.2.1. Basics of SIMT"
section: "2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#basics-of-simt"
---

## [2.2.1. Basics of SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#basics-of-simt)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#basics-of-simt "Permalink to this headline")

From the developer’s perspective, the CUDA thread is the fundamental unit of parallelism.  [Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt) describes the basic SIMT model of GPU execution and [SIMT Execution Model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-simt-architecture) provides additional details of the SIMT model.  The SIMT model allows each thread to maintain its own state and control flow. From a functional perspective, each thread can execute a separate code path.  However, substantial performance improvements can be realized by taking care that kernel code minimizes the situations where threads in the same warp take divergent code paths.
