---
title: "3.2.4. Advanced Synchronization Primitives"
section: "3.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-synchronization-primitives"
---

## [3.2.4. Advanced Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#advanced-synchronization-primitives)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-synchronization-primitives "Permalink to this headline")

This section introduces three families of synchronization primitives:

- [Scoped Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives-atomics), which pair C++ memory ordering with CUDA thread scopes to safely communicate across threads at block, cluster, device, or system scope (see [Thread Scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-thread-scopes)).
- [Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives-barriers), which split synchronization into arrival and wait phases, and can be used to track the progress of asynchronous operations.
- [Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives-pipelines), which stage work and coordinate multi-buffer producer–consumer patterns, commonly used to overlap compute with [asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-async-copies).
