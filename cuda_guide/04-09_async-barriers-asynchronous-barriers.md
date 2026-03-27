---
title: "4.9. Asynchronous Barriers"
section: "4.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#async-barriers--asynchronous-barriers"
---

# [4.9. Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#asynchronous-barriers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#asynchronous-barriers "Permalink to this headline")

Asynchronous barriers, introduced in [Advanced Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives), extend CUDA synchronization beyond `__syncthreads()` and `__syncwarp()`, enabling fine-grained, non-blocking coordination and better overlap of communication and computation.

This section provides details on how to use asynchronous barriers mainly via the `cuda::barrier` API (with pointers to `cuda::ptx` and primitives where applicable).
