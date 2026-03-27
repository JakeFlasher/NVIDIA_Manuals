---
title: "3.2.4.2. Asynchronous Barriers"
section: "3.2.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-barriers"
---

### [3.2.4.2. Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#asynchronous-barriers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#asynchronous-barriers "Permalink to this headline")

An asynchronous barrier differs from a typical single-stage barrier (`__syncthreads()`) in that the notification by a thread that it has reached the barrier (the “arrival”) is separated from the operation of waiting for other threads to arrive at the barrier (the “wait”). This separation increases execution efficiency by allowing a thread to perform additional operations unrelated to the barrier, making more efficient use of the wait time. Asynchronous barriers can be used to implement producer-consumer patterns with CUDA threads or enable asynchronous data copies within the memory hierarchy by having the copy operation signal (“arrive on”) a barrier upon completion.

Asynchronous barriers are available on devices of compute capability 7.0 or higher. Devices of compute capability 8.0 or higher provide hardware acceleration for asynchronous barriers in shared-memory and a significant advancement in synchronization granularity, by allowing hardware-accelerated synchronization of any subset of CUDA threads within the block. Previous architectures only accelerate synchronization at a whole-warp (`__syncwarp()`) or whole-block (`__syncthreads()`) level.

The CUDA programming model provides asynchronous barriers via `cuda::std::barrier`, an ISO C++-conforming barrier available in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html) library. In addition to implementing [std::barrier](https://en.cppreference.com/w/cpp/thread/barrier.html), the library offers CUDA-specific extensions to select a barrier’s thread scope to improve performance and exposes a lower-level [cuda::ptx](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) API. A `cuda::barrier` can interoperate with `cuda::ptx` by using the `friend` function `cuda::device::barrier_native_handle()` to retrieve the barrier’s native handle and pass it to `cuda::ptx` functions. CUDA also provides a [primitives API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#async-barriers-primitives-api) for asynchronous barriers in shared memory at thread-block scope.

The following table gives an overview of asynchronous barriers available for synchronizing at different thread scopes.

> | Thread Scope | Memory Location | Arrive on Barrier | Wait on Barrier | Hardware-accelerated | CUDA APIs |
> | --- | --- | --- | --- | --- | --- |
> | block | local shared memory | allowed | allowed | yes (8.0+) | `cuda::barrier`, `cuda::ptx`, primitives |
> | cluster | local shared memory | allowed | allowed | yes (9.0+) | `cuda::barrier`, `cuda::ptx` |
> | cluster | remote shared memory | allowed | not allowed | yes (9.0+) | `cuda::barrier`, `cuda::ptx` |
> | device | global memory | allowed | allowed | no | `cuda::barrier` |
> | system | global/unified memory | allowed | allowed | no | `cuda::barrier` |

**Temporal Splitting of Synchronization**

Without the asynchronous arrive-wait barriers, synchronization within a thread block is achieved using `__syncthreads()` or `block.sync()` when using [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups).

```c++
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */

         // Wait for all threads to arrive here.
        block.sync();

        /* code after wait */
    }
}
```

Threads are blocked at the synchronization point (`block.sync()`) until all threads have reached the synchronization point. In addition, memory updates that happened before the synchronization point are guaranteed to be visible to all threads in the block after the synchronization point.

This pattern has three stages:

- Code **before** the sync performs memory updates that will be read **after** the sync.
- Synchronization point.
- Code **after** the sync, with visibility of memory updates that happened **before** the sync.

Using asynchronous barriers instead, the temporally-split synchronization pattern is as follows.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   using barrier_t = cuda::barrier<cuda::thread_scope_block>;   __shared__ barrier_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     barrier_t::arrival_token token = bar.arrive();      compute(data, i);      // Wait for all threads participating in the barrier to complete bar.arrive().     bar.wait(std::move(token));      /* code after wait */   } } ``` |
| --- |

**CUDA C++ cuda::ptx**

| ```cuda #include <cuda/ptx> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   __shared__ uint64_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     cuda::ptx::mbarrier_init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     uint64_t token = cuda::ptx::mbarrier_arrive(&bar);      compute(data, i);      // Wait for all threads participating in the barrier to complete mbarrier_arrive().     while(!cuda::ptx::mbarrier_try_wait(&bar, token)) {}      /* code after wait */   } } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   __shared__ __mbarrier_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     __mbarrier_init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     __mbarrier_token_t token = __mbarrier_arrive(&bar);      compute(data, i);      // Wait for all threads participating in the barrier to complete __mbarrier_arrive().     while(!__mbarrier_try_wait(&bar, token, 1000)) {}      /* code after wait */   } } ``` |
| --- |

In this pattern, the synchronization point is split into an arrive point (`bar.arrive()`) and a wait point (`bar.wait(std::move(token))`). A thread begins participating in a `cuda::barrier` with its first call to `bar.arrive()`. When a thread calls `bar.wait(std::move(token))` it will be blocked until participating threads have completed `bar.arrive()` the expected number of times, which is the expected arrival count argument passed to `init()`. Memory updates that happen before participating threads’ call to `bar.arrive()` are guaranteed to be visible to participating threads after their call to `bar.wait(std::move(token))`. Note that the call to `bar.arrive()` does not block a thread, it can proceed with other work that does not depend upon memory updates that happen before other participating threads’ call to `bar.arrive()`.

The _arrive and wait_ pattern has five stages:

- Code **before** the arrive performs memory updates that will be read **after** the wait.
- Arrive point with implicit memory fence (i.e., equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`).
- Code **between** arrive and wait.
- Wait point.
- Code **after** the wait, with visibility of updates that were performed **before** the arrive.

For a comprehensive guide on how to use asynchronous barriers, see [Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers).
