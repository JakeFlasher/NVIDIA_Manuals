---
title: "4.3.5.1. cudaMemcpyAsync Current Context/Device Sensitivity"
section: "4.3.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudamemcpyasync-current-context-device-sensitivity"
---

### [4.3.5.1. cudaMemcpyAsync Current Context/Device Sensitivity](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamemcpyasync-current-context-device-sensitivity)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamemcpyasync-current-context-device-sensitivity "Permalink to this headline")

In the current CUDA driver, any async `memcpy` involving memory from
`cudaMallocAsync` should be done using the specified stream’s context as the
calling thread’s current context. This is not necessary for
`cudaMemcpyPeerAsync`, as the device primary contexts specified in the API
are referenced instead of the current context.
