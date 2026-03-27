---
title: "3.2.3. Thread Scopes"
section: "3.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scopes"
---

## [3.2.3. Thread Scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#thread-scopes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#thread-scopes "Permalink to this headline")

CUDA threads form a [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-thread-hierarchy-review), and using this hierarchy is essential for writing both correct and performant CUDA kernels. Within this hierarchy, the visibility and synchronization scope of memory operations can vary. To account for this non-uniformity, the CUDA programming model introduces the concept of _thread scopes_. A thread scope defines which threads can observe a thread’s loads and stores and specifies which threads can synchronize with each other using synchronization primitives such as atomic operations and barriers. Each scope has an associated point of coherency in the memory hierarchy.

Thread scopes are exposed in [CUDA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=thread%2520scopes#scope) and are also available as extensions in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes) library. The following table defines the thread scopes available:

| CUDA C++ Thread Scope | CUDA PTX Thread Scope | Description | Point of Coherency in Memory Hierarchy |
| --- | --- | --- | --- |
| `cuda::thread_scope_thread` |  | Memory operations are visible only to the local thread. | – |
| `cuda::thread_scope_block` | `.cta` | Memory operations are visible to other threads in the same thread block. | L1 |
|  | `.cluster` | Memory operations are visible to other threads in the same thread block cluster. | L2 |
| `cuda::thread_scope_device` | `.gpu` | Memory operations are visible to other threads in the same GPU device. | L2 |
| `cuda::thread_scope_system` | `.sys` | Memory operations are visible to other threads in the same system (CPU, other GPUs). | L2 + connected caches |

Sections [Advanced Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives) and [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-async-copies) demonstrate use of thread scopes.
