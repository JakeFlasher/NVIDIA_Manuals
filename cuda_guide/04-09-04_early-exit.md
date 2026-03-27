---
title: "4.9.4. Early Exit"
section: "4.9.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#early-exit"
---

## [4.9.4. Early Exit](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#early-exit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#early-exit "Permalink to this headline")

When a thread that is participating in a sequence of synchronizations must exit early from that sequence, that thread must explicitly drop out of participation before exiting. The remaining participating threads can proceed normally with subsequent arrive and wait operations.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier> #include <cooperative_groups.h>  __device__ bool condition_check();  __global__ void early_exit_kernel(int N) {   __shared__ cuda::barrier<cuda::thread_scope_block> bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     init(&bar, block.size());   }   block.sync();    for (int i = 0; i < N; ++i)   {     if (condition_check())     {       bar.arrive_and_drop();       return;     }     // Other threads can proceed normally.     auto token = bar.arrive();      /* code between arrive and wait */      // Wait for all threads to arrive.     bar.wait(std::move(token));      /* code after wait */   } } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h>  __device__ bool condition_check();  __global__ void early_exit_kernel(int N) {   __shared__ __mbarrier_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     __mbarrier_init(&bar, block.size());   }   block.sync();    for (int i = 0; i < N; ++i)   {     if (condition_check())     {       __mbarrier_token_t token = __mbarrier_arrive_and_drop(&bar);       return;     }     // Other threads can proceed normally.     __mbarrier_token_t token = __mbarrier_arrive(&bar);      /* code between arrive and wait */      // Wait for all threads to arrive.     while (!__mbarrier_try_wait(&bar, token, 1000)) {}      /* code after wait */   } } ``` |
| --- |

The `bar.arrive_and_drop()` operation arrives on the barrier to fulfill the participating thread’s obligation to arrive in the **current** phase, and then decrements the expected arrival count for the **next** phase so that this thread is no longer expected to arrive on the barrier.
