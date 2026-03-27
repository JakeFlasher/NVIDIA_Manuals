---
title: "4.9.1. Initialization"
section: "4.9.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#initialization"
---

## [4.9.1. Initialization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#initialization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#initialization "Permalink to this headline")

Initialization must happen before any thread begins participating in a barrier.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier> #include <cooperative_groups.h>  __global__ void init_barrier() {   __shared__ cuda::barrier<cuda::thread_scope_block> bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // A single thread initializes the total expected arrival count.     init(&bar, block.size());   }   block.sync(); } ``` |
| --- |

**CUDA C++ cuda::ptx**

| ```cuda #include <cuda/ptx> #include <cooperative_groups.h>  __global__ void init_barrier() {   __shared__ uint64_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // A single thread initializes the total expected arrival count.     cuda::ptx::mbarrier_init(&bar, block.size());   }   block.sync(); } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h>  __global__ void init_barrier() {   __shared__ uint64_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // A single thread initializes the total expected arrival count.     __mbarrier_init(&bar, block.size());   }   block.sync(); } ``` |
| --- |

Before any thread can participate in a barrier, the barrier must be initialized using the `cuda::barrier::init()` friend function. This must happen before any thread arrives on the barrier. This poses a bootstrapping challenge in that threads must synchronize before participating in the barrier, but threads are creating a barrier in order to synchronize. In this example, threads that will participate are part of a cooperative group and use `block.sync()` to bootstrap initialization. Since a whole thread block is participating in the barrier, `__syncthreads()` could also be used.

The second parameter of `init()` is the _expected arrival count_, i.e., the number of times `bar.arrive()` will be called by participating threads before a participating thread is unblocked from its call to `bar.wait(std::move(token))`. In this and the previous examples, the barrier is initialized with the number of threads in the thread block i.e., `cooperative_groups::this_thread_block().size()`, so that all threads within the thread block can participate in the barrier.

Asynchronous barriers are flexible in specifying _how_ threads participate (split arrive/wait) and _which_ threads participate. In contrast, `this_thread_block.sync()` or `__syncthreads()` is applicable to the whole thread-block and `__syncwarp(mask)` to a specified subset of a warp. Nonetheless, if the intention of the user is to synchronize a full thread block or a full warp, we recommend using `__syncthreads()` and `__syncwarp()` respectively for better performance.
