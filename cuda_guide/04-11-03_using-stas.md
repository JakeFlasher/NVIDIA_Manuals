---
title: "4.11.3. Using STAS"
section: "4.11.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-stas"
---

## [4.11.3. Using STAS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-stas)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-stas "Permalink to this headline")

CUDA applications using [thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters) may need to move small data elements between thread blocks within the cluster. STAS instructions (CC 9.0+, see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)) enable asynchronous data copies directly from registers to distributed shared memory. STAS is only exposed through a lower-level `cuda::ptx::st_async` API available in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/st_async.html?highlight=st_async#) library.

**Dimensions**. STAS supports copying 4, 8 or 16 bytes.

**Source and destination**. The only direction supported for asynchronous copy operations with STAS is from registers to distributed shared memory. The destination pointer needs to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied.

**Asynchronicity**. Data transfers using STAS are [asynchronous](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async thread operations (see [Async Thread and Async Proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. _Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future_. The completion mechanisms that STAS operations can use to signal that they have completed are [shared memory barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers).

In the following example, we show how to use STAS to implement a producer-consumer pattern within a thread-block cluster. This kernel creates a circular communication pipeline where 8 thread blocks are arranged in a ring, and each block simultaneously:

- Produces data for the next block in the sequence.
- Consumes data from the previous block in the sequence.

To implement this pattern, we need 2 shared memory barriers per thread block, one to notify the consumer block that the data has been copied to the shared memory buffer (`filled`) and one to notify the producer block that the buffer on the consumer is ready to be filled (`ready`).

**CUDA C++ cuda::ptx**

| ```cuda #include <cooperative_groups.h> #include <cuda/barrier> #include <cuda/ptx>  __global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel()  {     using namespace cooperative_groups;     using namespace cuda::device;     using namespace cuda::ptx;     using barrier_t = cuda::barrier<cuda::thread_scope_block>;      auto cluster = this_cluster();      #pragma nv_diag_suppress static_var_with_dynamic_init     __shared__ int buffer[BLOCK_SIZE];     __shared__ barrier_t filled;     __shared__ barrier_t ready;          // Initialize shared memory barriers.     if (threadIdx.x == 0) {         init(&filled, 1);         init(&ready, BLOCK_SIZE);     }          // Sync cluster to ensure remote barriers are initialized.     cluster.sync();          // Define my own and my neighbor's ranks.     int rk = cluster.block_rank();     int rk_next = (rk + 1) % 8;     int rk_prev = (rk + 7) % 8;            // Get addresses of remote buffer we are writing to and remote barriers of previous and next blocks.     auto buffer_next = cluster.map_shared_rank(buffer, rk_next);     auto bar_next = cluster.map_shared_rank(barrier_native_handle(filled), rk_next);     auto bar_prev = cluster.map_shared_rank(barrier_native_handle(ready), rk_prev);          int phase = 0;     for (int it = 0; it < 1000; ++it) {                  // As producers, send data to our right neighbor.         st_async(&buffer_next[threadIdx.x], rk, bar_next);                  if (threadIdx.x == 0) {             // Thread 0 arrives on local barrier and indicates it expects to receive a certain number of bytes.             mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, barrier_native_handle(filled), sizeof(buffer));         }          // As consumers, wait on local barrier for data from left neighbor to arrive.         while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase, 1000)) {}                  // At this point, the data has been copied to our local buffer.         int r = buffer[threadIdx.x];                  // Use the data to do something.          // As consumers, notify our left neighbor that we are done with the data.         mbarrier_arrive(sem_release, scope_cluster, space_cluster, bar_prev);                  // As producers, wait on local barrier until the right neighbor is ready to receive new data.         while (!mbarrier_try_wait_parity(barrier_native_handle(ready), phase, 1000)) {}         phase ^= 1;     } } ``` |
| --- |

- Shared memory barriers are initialized by the first thread of each block. Barrier `filled` is initialized to 1 and barrier `ready` is initialized to the number of threads in the block.
- A cluster-wide synchronization is performed to ensure that all barriers are initialized before any thread starts communication.
- Each thread determines its neighbors’ ranks and uses them to map the remote shared memory barriers and the remote shared memory buffer to write data to.
- In each iteration:
  1. As a producer, each thread sends data to its right neighbor.
  2. As a consumer, thread 0 arrives on the local `filled` barrier and indicates it expects to receive a certain number of bytes.
  3. As a consumer, each thread waits on the local `filled` barrier for data from the left neighbor to arrive.
  4. As a consumer, each thread uses the data to do something.
  5. As a consumer, each thread notifies the left neighbor that it is done with the data.
  6. As a producer, each thread waits on the local `ready` barrier until the right neighbor is ready to receive new data.

Note that for each barrier, we need to use the correct space. For mapped remote barriers, we need to use the `space_cluster` space, while for local barriers, we need to use the `space_shared` space.
