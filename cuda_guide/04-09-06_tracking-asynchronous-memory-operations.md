---
title: "4.9.6. Tracking Asynchronous Memory Operations"
section: "4.9.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#tracking-asynchronous-memory-operations"
---

## [4.9.6. Tracking Asynchronous Memory Operations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#tracking-asynchronous-memory-operations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#tracking-asynchronous-memory-operations "Permalink to this headline")

Asynchronous barriers can be used to track [asynchronous memory copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies). When an asynchronous copy operation is bound to a barrier, the copy operation automatically increments the expected count of the current barrier phase upon initiation and decrements it upon completion. This mechanism ensures that the barrier’s `wait()` operation will block until all associated asynchronous memory copies have completed, providing a convenient way to synchronize multiple concurrent memory operations.

Starting with compute capability 9.0, asynchronous barriers in shared memory with thread-block or cluster scope can **explicitly** track asynchronous memory operations. We refer to these barriers as _asynchronous transaction barriers_. In addition to the expected arrival count, a barrier object can accept a **transaction count**, which can be used for tracking the completion of asynchronous transactions. The transaction count tracks the number of asynchronous transactions that are outstanding and yet to be complete, in units specified by the asynchronous memory operation (typically bytes). The transaction count to be tracked by the current phase can be set on arrival with `cuda::device::barrier_arrive_tx()` or directly with `cuda::device::barrier_expect_tx()`. When a barrier uses a transaction count, it blocks threads at the wait operation until all the producer threads have performed an arrive _and_ the sum of all the transaction counts reaches an expected value.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier> #include <cooperative_groups.h>  __global__ void track_kernel() {   __shared__ cuda::barrier<cuda::thread_scope_block> bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     init(&bar, block.size());   }   block.sync();    auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);    bar.wait(cuda::std::move(token)); } ``` |
| --- |

**CUDA C++ cuda::ptx**

| ```cuda #include <cuda/ptx> #include <cooperative_groups.h>  __global__ void track_kernel() {   __shared__ uint64_t bar;   auto block = cooperative_groups::this_thread_block();    if (block.thread_rank() == 0)   {     cuda::ptx::mbarrier_init(&bar, block.size());   }   block.sync();    uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_shared, &bar, 1, 0);    while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {} } ``` |
| --- |

In this example, the `cuda::device::barrier_arrive_tx()` operation constructs an arrival token object associated with the phase synchronization point for the current phase. Then, decrements the arrival count by 1 and increments the expected transaction count by 0. Since the transaction count update is 0, the barrier is not tracking any transactions. The subsequent section on [Using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-tma) includes examples of tracking asynchronous memory operations.
