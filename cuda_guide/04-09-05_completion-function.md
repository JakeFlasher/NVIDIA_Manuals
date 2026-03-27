---
title: "4.9.5. Completion Function"
section: "4.9.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#completion-function"
---

## [4.9.5. Completion Function](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#completion-function)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#completion-function "Permalink to this headline")

The `cuda::barrier` API supports an optional completion function. A `CompletionFunction` of `cuda::barrier<Scope, CompletionFunction>` is executed once per phase, after the last thread _arrives_ and before any thread is unblocked from the `wait`. Memory operations performed by the threads that arrived at the `barrier` during the phase are visible to the thread executing the `CompletionFunction`, and all memory operations performed within the `CompletionFunction` are visible to all threads waiting at the `barrier` once they are unblocked from the `wait`.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier> #include <cooperative_groups.h> #include <functional> namespace cg = cooperative_groups;  __device__ int divergent_compute(int *, int); __device__ int independent_computation(int *, int);  __global__ void psum(int *data, int n, int *acc) {   auto block = cg::this_thread_block();    constexpr int BlockSize = 128;   __shared__ int smem[BlockSize];   assert(BlockSize == block.size());   assert(n % BlockSize == 0);    auto completion_fn = [&]   {     int sum = 0;     for (int i = 0; i < BlockSize; ++i)     {       sum += smem[i];     }     *acc += sum;   };    /* Barrier storage.      Note: the barrier is not default-constructible because            completion_fn is not default-constructible due            to the capture. */   using completion_fn_t = decltype(completion_fn);   using barrier_t = cuda::barrier<cuda::thread_scope_block,                                   completion_fn_t>;   __shared__ std::aligned_storage<sizeof(barrier_t),                                   alignof(barrier_t)>       bar_storage;    // Initialize barrier.   barrier_t *bar = (barrier_t *)&bar_storage;   if (block.thread_rank() == 0)   {     assert(*acc == 0);     assert(blockDim.x == blockDim.y == blockDim.y == 1);     new (bar) barrier_t{block.size(), completion_fn};     /* equivalent to: init(bar, block.size(), completion_fn); */   }   block.sync();    // Main loop.   for (int i = 0; i < n; i += block.size())   {     smem[block.thread_rank()] = data[i] + *acc;     auto token = bar->arrive();     // We can do independent computation here.     bar->wait(std::move(token));     // Shared-memory is safe to re-use in the next iteration     // since all threads are done with it, including the one     // that did the reduction.   } } ``` |
| --- |
