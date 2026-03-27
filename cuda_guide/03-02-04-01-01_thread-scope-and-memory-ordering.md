---
title: "3.2.4.1.1. Thread Scope and Memory Ordering"
section: "3.2.4.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scope-and-memory-ordering"
---

#### [3.2.4.1.1. Thread Scope and Memory Ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#thread-scope-and-memory-ordering)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#thread-scope-and-memory-ordering "Permalink to this headline")

Scoped atomics combine two key concepts:

- **Thread Scope**: defines which threads can observe the effect of the atomic operation (see [Thread Scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-thread-scopes)).
- **Memory Ordering**: defines the ordering constraints relative to other memory operations (see [C++ standard atomic memory semantics](https://en.cppreference.com/w/cpp/atomic/memory_order.html)).

**CUDA C++ cuda::atomic**

```cuda
#include <cuda/atomic>

__global__ void block_scoped_counter() {
    // Shared atomic counter visible only within this block
    __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

    // Initialize counter (only one thread should do this)
    if (threadIdx.x == 0) {
        counter.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // All threads in block atomically increment
    int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);

    // Use old_value...
}
```

**Built-in Atomic Functions**

```cuda
__global__ void block_scoped_counter() {
    // Shared counter visible only within this block
    __shared__ int counter;

    // Initialize counter (only one thread should do this)
    if (threadIdx.x == 0) {
        __nv_atomic_store_n(&counter, 0,
                            __NV_ATOMIC_RELAXED,
                            __NV_THREAD_SCOPE_BLOCK);
    }
    __syncthreads();

    // All threads in block atomically increment
    int old_value = __nv_atomic_fetch_add(&counter, 1,
                                          __NV_ATOMIC_RELAXED,
                                          __NV_THREAD_SCOPE_BLOCK);

    // Use old_value...
}
```

This example implements a _block-scoped atomic counter_ that demonstrates the fundamental concepts of scoped atomics:

- **Shared Variable**: a single counter is shared among all threads in the block using `__shared__` memory.
- **Atomic Type Declaration**: `cuda::atomic<int, cuda::thread_scope_block>` creates an atomic integer with block-level visibility.
- **Single Initialization**: only thread 0 initializes the counter to prevent race conditions during setup.
- **Block Synchronization**: `__syncthreads()` ensures all threads see the initialized counter before proceeding.
- **Atomic Increment**: each thread atomically increments the counter and receives the previous value.

`cuda::memory_order_relaxed` is chosen here because we only need atomicity (indivisible read-modify-write) without ordering constraints between different memory locations. Since this is a straightforward counting operation, the order of increments doesn’t matter for correctness.

For producer-consumer patterns, acquire-release semantics ensure proper ordering:

**CUDA C++ cuda::atomic**

```cuda
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

    if (threadIdx.x == 0) {
        // Producer: write data then signal ready
        data = 42;
        ready.store(true, cuda::memory_order_release);  // Release ensures data write is visible
    } else {
        // Consumer: wait for ready signal then read data
        while (!ready.load(cuda::memory_order_acquire)) {  // Acquire ensures data read sees the write
            // spin wait
        }
        int value = data;
        // Process value...
    }
}
```

**Built-in Atomic Functions**

```cuda
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ bool ready; // Only ready flag needs atomic operations

    if (threadIdx.x == 0) {
        // Producer: write data then signal ready
        data = 42;
        __nv_atomic_store_n(&ready, true,
                            __NV_ATOMIC_RELEASE,
                            __NV_THREAD_SCOPE_BLOCK);  // Release ensures data write is visible
    } else {
        // Consumer: wait for ready signal then read data
        while (!__nv_atomic_load_n(&ready,
                                   __NV_ATOMIC_ACQUIRE,
                                   __NV_THREAD_SCOPE_BLOCK)) {  // Acquire ensures data read sees the write
            // spin wait
        }
        int value = data;
        // Process value...
    }
}
```
