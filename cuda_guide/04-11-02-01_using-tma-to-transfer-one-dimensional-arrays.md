---
title: "4.11.2.1. Using TMA to transfer one-dimensional arrays"
section: "4.11.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-tma-to-transfer-one-dimensional-arrays"
---

### [4.11.2.1. Using TMA to transfer one-dimensional arrays](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-tma-to-transfer-one-dimensional-arrays)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-tma-to-transfer-one-dimensional-arrays "Permalink to this headline")

The following table summarizes the possible source and destination memory spaces and completion mechanisms for bulk-asynchronous TMA along with the API that exposes it.

| Source | Destination | Completion Mechanism | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global | bulk async-group | [cuda::ptx::cp_async_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| global | shared::cta | shared memory barrier | [cuda::memcpy_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html), [cuda::device::memcpy_async_tx](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async_tx.html), [cuda::ptx::cp_async_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| global | shared::cluster | shared memory barrier | [cuda::ptx::cp_async_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| shared::cta | shared::cluster | shared memory barrier | [cuda::ptx::cp_async_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| shared::cta | shared::cta |  |  |

Some functionality requires inline PTX that is currently made available through the `cuda::ptx` namespace in the [CUDA Standard C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) library. The availability of these wrappers can be checked with the following code:

```c++
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

Note that `cuda::memcpy_async` uses TMA if the source and destination addresses are 16-byte aligned and the size is a multiple of 16 bytes, otherwise it falls back to synchronous copies. On the other hand, `cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` always use TMA and will result in undefined behavior if the requirements are not met.

In the following, we demonstrate how to use bulk-asynchronous copies through an example. The example read-modify-writes a one-dimensional array. The kernel goes through the following steps:

1. Initialize a shared memory barrier as a completion mechanism for the bulk-asynchronous copy from global to shared memory.
2. Initiate the copy of a block of memory from global to shared memory.
3. Arrive and wait on the shared memory barrier for completion of the copy.
4. Increment the shared memory buffer values.
5. Use a proxy fence to ensure shared memory writes (generic proxy) become visible to the subsequent bulk-asynchronous copy (async proxy).
6. Initiate a bulk-asynchronous copy of the buffer in shared memory to global memory.
7. Wait for the bulk-asynchronous copy to have finished reading shared memory.

```cuda
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operation should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory from a single thread.
  if (is_elected()) {
    // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).

    // Version 1: cuda::memcpy_async
    cuda::memcpy_async(
        smem_data, data + offset,
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar);

    // Version 2: cuda::device::memcpy_async_tx
    // cuda::device::memcpy_async_tx(
    //   smem_data, data + offset,
    //   cuda::aligned_size_t<16>(sizeof(smem_data)),
    //   bar);
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));

    // Version 3: cuda::ptx::cp_async_bulk
    // ptx::cp_async_bulk(
    //     ptx::space_shared, ptx::space_global,
    //     smem_data, data + offset,
    //     sizeof(smem_data),
    //     cuda::device::barrier_native_handle(bar));
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));
  }

  // 3a. All threads arrive on the barrier.
  barrier::arrival_token token = bar.arrive();

  // 3b. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory.
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory.
  if (is_elected()) {
    ptx::cp_async_bulk(
        ptx::space_global, ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

**Barrier initialization**. The barrier is initialized with the number of threads participating in the block. As a result, the barrier will flip only if
all threads have arrived on this barrier. Shared memory barriers are described
in more detail in [shared memory barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers).

**TMA read**. The bulk-asynchronous copy instruction directs the
hardware to copy a large chunk of data into shared memory, and to update the
[transaction count](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)
of the shared memory barrier after completing the read. In general, issuing as
few bulk copies with as big a size as possible results in the best performance.
Because the copy can be performed asynchronously by the hardware, it is not
necessary to split the copy into smaller chunks.

The thread that initiates the bulk-asynchronous copy operation also tells the barrier how many transactions (tx) are expected to arrive.
In this case, the transactions are counted in bytes. This is automatically performed by `cuda::memcpy_async`, but not by
`cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` after which we need to explicitly call `cuda::ptx::mbarrier_expect_tx`.
If multiple threads update the transaction count, the expected transaction will be the sum
of the updates. The barrier will only flip once all threads have arrived **and**
all bytes have arrived. Once the barrier has flipped, the bytes are safe to read
from shared memory, both by the threads as well as by subsequent
bulk-asynchronous copies. More information about barrier transaction accounting
can be found in [Tracking Asynchronous Memory Operations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers-tracking).

**Barrier wait**. Waiting for the barrier to flip is done using tokens with `bar.wait()`. It can be more efficient to use explicit phase tracking of the barrier (see [Explicit Phase Tracking](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers-explicit-phase)).

**SMEM write and sync**. The increment of the buffer values reads and writes to shared
memory. To make the writes visible to subsequent bulk-asynchronous copies, the
`cuda::ptx::fence_proxy_async` function is used. This orders the writes to
shared memory before subsequent reads from bulk-asynchronous copy operations,
which read through the async proxy. So each thread first orders the writes to
objects in shared memory in the async proxy via the
`cuda::ptx::fence_proxy_async`, and these operations by all threads are
ordered before the async operation performed in thread 0 using
`__syncthreads()`.

**TMA write and sync**. The write from shared to global memory is again
initiated by a single thread. The completion of the write is not tracked by a
shared memory barrier. Instead, a thread-local mechanism is used. Multiple
writes can be batched into a so-called _bulk async-group_. Afterwards, the
thread can wait for all operations in this group to have completed reading from
shared memory (as in the code above) or to have completed writing to global
memory, making the writes visible to the initiating thread. For more information,
refer to the PTX ISA documentation of [cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
Note that the bulk-asynchronous and non-bulk-asynchronous copy instructions have
different async-groups: there exist both `cp.async.wait_group` and
`cp.async.bulk.wait_group` instructions.

> **Note**
>
> It is recommended to initiate TMA operations by a single thread in the block.
> While using `if (threadIdx.x == 0)` might seem sufficient, the compiler cannot
> verify that indeed only one thread is initiating the copy and may insert a peeling
> loop over all active threads, which results in warp serialization and reduced
> performance. To prevent this, we define the `is_elected()` helper function that
> uses `cuda::ptx::elect_sync` to select one thread from warp 0 – which is known to
> the compiler – to execute the copy allowing it to generate more efficient code.
> Alternatively, the same effect can be achieved with [cooperative_groups::invoke_one](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups-invoke-one).

The bulk-asynchronous instructions have specific alignment requirements on their source and
destination addresses. More information can be found in the table below.

| Address / Size | Alignment |
| --- | --- |
| Global memory address | Must be 16 byte aligned. |
| Shared memory address | Must be 16 byte aligned. |
| Shared memory barrier address | Must be 8 byte aligned (this is guaranteed by `cuda::barrier`). |
| Size of transfer | Must be a multiple of 16 bytes. |
