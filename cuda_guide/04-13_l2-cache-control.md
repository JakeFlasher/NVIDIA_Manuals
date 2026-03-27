---
title: "4.13. L2 Cache Control"
section: "4.13"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#l2-cache-control"
---

# [4.13. L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#l2-cache-control)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#l2-cache-control "Permalink to this headline")

When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be persisting. On the other hand, if the data is only accessed once, such data accesses can be considered to be streaming.

Devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory.

This functionality is exposed through two main APIs:

- The CUDA runtime API (starting with CUDA 11.0) provides programmatic control over L2 cache persistence.
- The `cuda::annotated_ptr` API in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html) library (starting with CUDA 11.5) annotates pointers in CUDA kernels with memory access properties to achieve a similar effect..

The following sections focus on the CUDA runtime API. For detailed information about the `cuda::annotated_ptr` approach, please refer to the [libcu++ documentation](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html).
