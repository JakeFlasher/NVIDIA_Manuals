---
title: "4.10. Pipelines"
section: "4.10"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines--pipelines"
---

# [4.10. Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#pipelines)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#pipelines "Permalink to this headline")

Pipelines, introduced in [Advanced Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives), are a mechanism for staging work and coordinating multi-buffer producer–consumer patterns, commonly used to overlap compute with [asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies).

This section provides details on how to use pipelines mainly via the `cuda::pipeline` API (with pointers to primitives where applicable).
