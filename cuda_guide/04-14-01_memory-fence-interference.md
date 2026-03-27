---
title: "4.14.1. Memory Fence Interference"
section: "4.14.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html#memory-fence-interference"
---

## [4.14.1. Memory Fence Interference](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-fence-interference)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-fence-interference "Permalink to this headline")

Some CUDA applications may see degraded performance due to memory fence/flush operations waiting on more transactions than those necessitated by the CUDA memory consistency model.

| ```c++ __managed__ int x = 0; __device__  cuda::atomic<int, cuda::thread_scope_device> a(0); __managed__ cuda::atomic<int, cuda::thread_scope_system> b(0); ``` |  |  |
| --- | --- | --- |
| Thread 1 (SM)    ```c++ x = 1; a = 1; ``` | Thread 2 (SM)    ```c++ while (a != 1) ; assert(x == 1); b = 1; ``` | Thread 3 (CPU)    ```c++ while (b != 1) ; assert(x == 1); ``` |

Consider the example above. The CUDA memory consistency model guarantees that the asserted condition will be true, so the write to `x` from thread 1 must be visible to thread 3, before the write to `b` from thread 2.

The memory ordering provided by the release and acquire of `a` is only sufficient to make `x` visible to thread 2, not thread 3, as it is a device-scope operation. The system-scope ordering provided by release and acquire of `b`, therefore, needs to ensure not only writes issued from thread 2 itself are visible to thread 3, but also writes from other threads that are visible to thread 2. This is known as cumulativity. As the GPU cannot know at the time of execution which writes have been guaranteed at the source level to be visible and which are visible only by chance timing, it must cast a conservatively wide net for in-flight memory operations.

This sometimes leads to interference: because the GPU is waiting on memory operations it is not required to at the source level, the fence/flush may take longer than necessary.

Note that fences may occur explicitly as intrinsics or atomics in code, like in the example, or implicitly to implement _synchronizes-with_ relationships at task boundaries.

A common example is when a kernel is performing computation in local GPU memory, and a parallel kernel (e.g. from NCCL) is performing communications with a peer. Upon completion, the local kernel will implicitly flush its writes to satisfy any _synchronizes-with_ relationships to downstream work. This may unnecessarily wait, fully or partially, on slower nvlink or PCIe writes from the communication kernel.
