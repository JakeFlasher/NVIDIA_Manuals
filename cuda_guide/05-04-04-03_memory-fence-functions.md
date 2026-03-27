---
title: "5.4.4.3. Memory Fence Functions"
section: "5.4.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-fence-functions"
---

### [5.4.4.3. Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#memory-fence-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#memory-fence-functions "Permalink to this headline")

The CUDA programming model assumes a weakly ordered memory model. In other words, the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or the memory of a peer device is not necessarily the order in which another CUDA or host thread observes the data being written. Reading from or writing to the same memory location without memory fences or synchronization results in undefined behavior.

In the following example, thread 1 executes `writeXY()`, while thread 2 executes `readXY()`.

```cuda
__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    int A = X;
}
```

The two threads simultaneously read and write to the same memory locations, `X` and `Y`. Any data race results in undefined behavior and has no defined semantics. Therefore, the resulting values for `A` and `B` can be anything.

Memory fence and synchronization functions enforce a [sequentially consistent ordering](https://en.cppreference.com/w/cpp/atomic/memory_order) of memory accesses. These functions differ in the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes) in which orderings are enforced, but are independent of the accessed memory space, including shared memory, global memory, page-locked host memory, and the memory of a peer device.

> **Hint**
>
> It is suggested to use `cuda::atomic_thread_fence` provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)  whenever possible for safety and portability reasons.

**Block-level memory fence**

**CUDA C++**

```cuda
// <cuda/atomic> header
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
```

ensures that:

- All writes to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are observed by all threads in the calling thread’s block as occurring before all writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`;
- All reads from all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are ordered before all reads from all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`.

**Intrinsics**

```cuda
void __threadfence_block();
```

ensures that:

- All writes to all memory made by the calling thread before the call to `__threadfence_block()` are observed by all threads in the calling thread’s block as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_block()`;
- All reads from all memory made by the calling thread before the call to `__threadfence_block()` are ordered before all reads from all memory made by the calling thread after the call to `__threadfence_block()`.

**Device-level memory fence**

**CUDA C++**

```cuda
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
```

ensures that:

- No writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()`.

**Intrinsics**

```cuda
void __threadfence();
```

ensures that:

- No writes to all memory made by the calling thread after the call to `__threadfence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `__threadfence()`.

**System-level memory fence**

**CUDA C++**

```cuda
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
```

ensures that:

- All writes to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`.

**Intrinsics**

```cuda
void __threadfence_system();
```

ensures that:

- All writes to all memory made by the calling thread before the call to `__threadfence_system()` are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_system()`.

In the previous code sample, we can insert memory fences in the code as follows:

**CUDA C++**

```cuda
#include <cuda/atomic>

__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    int A = X;
}
```

**Intrinsics**

```cuda
__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    __threadfence();
    int A = X;
}
```

For this code, the following outcomes can be observed:

- `A` equal to 1 and `B` equal to 2, namely `readXY()` is executed before `writeXY()`,
- `A` equal to 10 and `B` equal to 20, namely `writeXY()` is executed before `readXY()`.
- `A` equal to 10 and `B` equal to 2.
- The case where `A` is 1 and `B` is 20 is not possible, as the memory fence ensures that the write to `X` is visible before the write to `Y`.

If threads 1 and 2 belong to the same block, it is enough to use a block-level fence. If threads 1 and 2 do not belong to the same block, a device-level fence must be used if they are CUDA threads from the same device, and a system-level fence must be used if they are CUDA threads from two different devices.

A common use case is illustrated by the following code sample, where threads consume data produced by other threads. This kernel computes the sum of an array of N numbers in a single call.

- Each block first sums a subset of the array and stores the result in global memory.
- When all the blocks have finished, the last block reads each of these partial sums from global memory and adds them together to obtain the final result.
- To determine which block finished last, each block atomically increments a counter to signal completion of computing and storing its partial sum (see the [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomic-functions) section for further details). The last block receives a counter value equal to `gridDim.x - 1`.

Without a fence between storing the partial sum and incrementing the counter, the counter may increment before the partial sum is stored. This could cause the counter to reach `gridDim.x - 1` and allow the last block to start reading partial sums before they are updated in memory.

> **Note**
>
> The memory fence only affects the order in which memory operations are executed; it does not guarantee visibility of these operations to other threads.

In the code sample below, the visibility of the memory operations on the `result` variable is ensured by declaring it as `volatile`. For more details, see the `volatile`-[qualified variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#volatile-qualifier) section.

```cuda
#include <cuda/atomic>

__device__ int count = 0;

__global__ void sum(const float*    array,
                    int             N,
                    volatile float* result) {
    __shared__ bool isLastBlockDone;
    // Each block sums a subset of the input array.
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {
        // Thread 0 of each block stores the partial sum to global memory.
        // The compiler will use a store operation that bypasses the L1 cache
        // since the "result" variable is declared as volatile.
        // This ensures that the threads of the last block will read the correct
        // partial sums computed by all other blocks.
        result[blockIdx.x] = partialSum;

        // Thread 0 makes sure that the increment of the "count" variable is
        // only performed after the partial sum has been written to global memory.
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

        // Thread 0 signals that it is done.
        int count_old = atomicInc(&count, gridDim.x);

        // Thread 0 determines if its block is the last block to be done.
        isLastBlockDone = (count_old == (gridDim.x - 1));
    }
    // Synchronize to make sure that each thread reads the correct value of
    // isLastBlockDone.
    __syncthreads();

    if (isLastBlockDone) {
        // The last block sums the partial sums stored in result[0 .. gridDim.x-1]
        float totalSum = calculateTotalSum(result);

        if (threadIdx.x == 0) {
            // Thread 0 of last block stores the total sum to global memory and
            // resets the count variable, so that the next kernel call works
            // properly.
            result[0] = totalSum;
            count     = 0;
        }
    }
}
```
