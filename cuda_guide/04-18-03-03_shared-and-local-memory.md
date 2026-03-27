---
title: "4.18.3.3. Shared and Local Memory"
section: "4.18.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#shared-and-local-memory"
---

### [4.18.3.3. Shared and Local Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#shared-and-local-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#shared-and-local-memory "Permalink to this headline")

Shared and Local memory is private to a thread block or thread, respectively, and is not visible or coherent between parent and child. Behavior is undefined when an object in one of these locations is referenced outside of the scope within which it belongs, and may cause an error.

The NVIDIA compiler will attempt to warn if it can detect that a pointer to local or shared memory is being passed as an argument to a kernel launch. At runtime, the programmer may use the `__isGlobal()` intrinsic to determine whether a pointer references global memory and so may safely be passed to a child launch.

Calls to `cudaMemcpy*Async()` or `cudaMemset*Async()` may invoke new child kernels on the device in order to preserve stream semantics. As such, passing shared or local memory pointers to these APIs is illegal and will return an error.
