---
title: "4.18.5.2.1.2. Pending Kernel Launches"
section: "4.18.5.2.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#pending-kernel-launches"
---

##### [4.18.5.2.1.2. Pending Kernel Launches](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#pending-kernel-launches)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#pending-kernel-launches "Permalink to this headline")

When a kernel is launched, all associated configuration and parameter data is tracked until the kernel completes. This data is stored within a system-managed launch pool.

The size of the fixed-size launch pool is configurable by calling `cudaDeviceSetLimit()` from the host and specifying `cudaLimitDevRuntimePendingLaunchCount`.
