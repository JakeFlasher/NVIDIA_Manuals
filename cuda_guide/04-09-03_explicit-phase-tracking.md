---
title: "4.9.3. Explicit Phase Tracking"
section: "4.9.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#explicit-phase-tracking"
---

## [4.9.3. Explicit Phase Tracking](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#explicit-phase-tracking)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#explicit-phase-tracking "Permalink to this headline")

An asynchronous barrier can have multiple phases depending on how many times it is used to synchronize threads and memory operations. Instead of using tokens to track barrier phase flips, we can directly track a phase using the `mbarrier_try_wait_parity()` family of functions available through the `cuda::ptx` and primitives APIs.

In its simplest form, the `cuda::ptx::mbarrier_try_wait_parity(uint64_t* bar, const uint32_t& phaseParity)` function waits for a phase with a particular parity. The `phaseParity` operand is the integer parity of either the current phase or the immediately preceding phase of the barrier object. An even phase has integer parity 0 and an odd phase has integer parity 1. When we initialize a barrier, its phase has parity 0. So the valid values of `phaseParity` are 0 and 1. Explicit phase tracking can be useful when tracking [asynchronous memory operations](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies), as it allows only a single thread to arrive on the barrier and set the transaction count, while other threads only wait for a parity-based phase flip. This can be more efficient than having all threads arrive on the barrier and use tokens. This functionality is only available for shared-memory barriers at thread-block and cluster scope.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/ptx> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   using barrier_t = cuda::barrier<cuda::thread_scope_block>;   __shared__ barrier_t bar;   int parity = 0; // Initial phase parity is 0.   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     // Get a handle to the native barrier to use with cuda::ptx API.     (void)cuda::ptx::mbarrier_arrive(cuda::device::barrier_native_handle(bar));      compute(data, i);      // Wait for all threads participating in the barrier to complete mbarrier_arrive().     // Get a handle to the native barrier to use with cuda::ptx API.     while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), parity)) {}     // Flip parity.     parity ^= 1;      /* code after wait */   } } ``` |
| --- |

**CUDA C++ cuda::ptx**

| ```cuda #include <cuda/ptx> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   __shared__ uint64_t bar;   int parity = 0; // Initial phase parity is 0.   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     cuda::ptx::mbarrier_init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     (void)cuda::ptx::mbarrier_arrive(&bar);      compute(data, i);      // Wait for all threads participating in the barrier to complete mbarrier_arrive().     while (!cuda::ptx::mbarrier_try_wait_parity(&bar, parity)) {}     // Flip parity.     parity ^= 1;      /* code after wait */   } } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h>  __device__ void compute(float *data, int iteration);  __global__ void split_arrive_wait(int iteration_count, float *data) {   __shared__ __mbarrier_t bar;   bool parity = false; // Initial phase parity is false.   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     // Initialize barrier with expected arrival count.     __mbarrier_init(&bar, block.size());   }   block.sync();    for (int i = 0; i < iteration_count; ++i)   {     /* code before arrive */      // This thread arrives. Arrival does not block a thread.     (void)__mbarrier_arrive(&bar);      compute(data, i);      // Wait for all threads participating in the barrier to complete __mbarrier_arrive().     while(!__mbarrier_try_wait_parity(&bar, parity, 1000)) {}     parity ^= 1;      /* code after wait */   } } ``` |
| --- |
