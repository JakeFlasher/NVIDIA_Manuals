---
title: "4.18.6. Device-side Launch from PTX"
section: "4.18.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#device-side-launch-from-ptx"
---

## [4.18.6. Device-side Launch from PTX](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-side-launch-from-ptx)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-side-launch-from-ptx "Permalink to this headline")

The previous sections have discussed using the [CUDA Device Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cuda-device-runtime) to achieve dynamic parallelism. Dynamic parallelism can also be performed from PTX. For the programming language and compiler implementers who target _Parallel Thread Execution_ (PTX) and plan to support _Dynamic Parallelism_ in their language, this section provides the low-level details related to supporting kernel launches at the PTX level.
