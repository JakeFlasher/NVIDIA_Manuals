---
title: "4.11.1.1. Batching Loads in Conditional Code"
section: "4.11.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#batching-loads-in-conditional-code"
---

### [4.11.1.1. Batching Loads in Conditional Code](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#batching-loads-in-conditional-code)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#batching-loads-in-conditional-code "Permalink to this headline")

In this stencil example, the first warp of the thread block is responsible for collectively loading all the required data from the center as well as the left and right halos. With synchronous copies, due to the conditional nature of the code, the compiler may choose to generate a sequence of load-from-global (LDG) store-to-shared (STS) instructions instead of 3 LDGs followed by 3 STSs, which would be the optimal way to load the data to hide the global memory latency.

```c++
__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    // Left halo (8 elements) - center (32 elements) - right halo (8 elements)
    __shared__ float buffer[8 + 32 + 8];
    const int tid = threadIdx.x;

    if (tid < 8) {
        buffer[tid] = left[tid]; // Left halo
    } else if (tid >= 32 - 8) {
        buffer[tid + 16] = right[tid]; // Right halo
    }
    if (tid < 32) {
      buffer[tid + 8] = center[tid]; // Center
    }
    __syncthreads();

    // Compute stencil
}
```

To ensure that the data is loaded in the optimal way, we can replace the synchronous memory copies with asynchronous copies that load data directly from global memory to shared memory. This not only reduces register usage by copying the data directly to shared memory, but also ensures all loads from global memory are in-flight.

**CUDA C++ cuda::memcpy_async**

| ```cuda #include <cooperative_groups.h> #include <cuda/barrier>  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     auto block = cooperative_groups::this_thread_block();     auto thread = cooperative_groups::this_thread();     using barrier_t = cuda::barrier<cuda::thread_scope_block>;     __shared__ barrier_t barrier;     __shared__ float buffer[8 + 32 + 8];          // Initialize synchronization object.     if (block.thread_rank() == 0) {         init(&barrier, block.size());     }     __syncthreads();      // Version 1: Issue the copies in individual threads.     if (tid < 8) {         cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Left halo         // or cuda::memcpy_async(thread, buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     } else if (tid >= 32 - 8) {         cuda::memcpy_async(buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Right halo         // or cuda::memcpy_async(thread, buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     }     if (tid < 32) {         cuda::memcpy_async(buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Center         // or cuda::memcpy_async(thread, buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     }          // Version 2: Cooperatively issue the copies across all threads.     cuda::memcpy_async(block, buffer, left, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Left halo     cuda::memcpy_async(block, buffer + 8, center, cuda::aligned_size_t<4>(32 * sizeof(float)), barrier); // Center     cuda::memcpy_async(block, buffer + 40, right, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Right halo          // Wait for all copies to complete.     barrier.arrive_and_wait();     __syncthreads();      // Compute stencil       } ``` |
| --- |

**CUDA C++ cooperative_groups::memcpy_async**

| ```cuda #include <cooperative_groups.h> #include <cooperative_groups/memcpy_async.h>  namespace cg = cooperative_groups;  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     cg::thread_block block = cg::this_thread_block();     // Left halo (8 elements) - center (32 elements) - right halo (8 elements).      __shared__ float buffer[8 + 32 + 8];      // Cooperatively issue the copies across all threads.     cg::memcpy_async(block, buffer, left, 8 * sizeof(float)); // Left halo     cg::memcpy_async(block, buffer + 8, center, 32 * sizeof(float)); // Center     cg::memcpy_async(block, buffer + 40, right, 8 * sizeof(float)); // Right halo     cg::wait(block); // Waits for all copies to complete.     __syncthreads();      // Compute stencil. } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_pipeline.h>  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     // Left halo (8 elements) - center (32 elements) - right halo (8 elements).     __shared__ float buffer[8 + 32 + 8];     const int tid = threadIdx.x;      if (tid < 8) {         __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float)); // Left halo     } else if (tid >= 32 - 8) {         __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float)); // Right halo     }     if (tid < 32) {         __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float)); // Center     }     __pipeline_commit();     __pipeline_wait_prior(0);     __syncthreads();      // Compute stencil. } ``` |
| --- |

The `cuda::memcpy_async` overload for `cuda::barrier` enables synchronizing asynchronous data transfers using an [asynchronous barrier](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers). This overload executes the copy operation as-if performed by another thread bound to the barrier by incrementing the expected count of the current phase on creation, and decrementing it on completion of the copy operation, such that the phase of the `barrier` will only advance when all threads participating in the barrier have arrived, and all `memcpy_async` bound to the current phase of the barrier have completed. We use a block-wide `barrier`, where all threads in the block participate, and merge the arrival and wait on the barrier with `arrive_and_wait`, since we do not perform any work between the phases.

Note that we can either use thread-level copies (version 1) or collective copies (version 2) to achieve the same result. In version 2, the API will automatically handle how the copies are done under the hood. In both versions, we use `cuda::aligned_size_t<4>()` to inform the compiler that the data is aligned to 4 bytes and the size of the data to copy is a multiple of 4 to enable use of LDGSTS. Note that for interoperability with `cuda::barrier`, `cuda::memcpy_async` from the `cuda/barrier` header is used here.

The [cooperative_groups::memcpy_async](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-async-memcpy) implementation coordinates the memory transfers collectively across all threads in the block, but synchronizes completion with `cg::wait(block)` instead of explicit barrier operations.

The implementation based on the low-level primitives uses `__pipeline_memcpy_async()` to initiate element-wise memory transfers, `__pipeline_commit()` to commit the batch of copies, and `__pipeline_wait_prior(0)` to wait for all operations in the pipeline to complete. This provides the most direct control at the expense of more verbose code compared to the higher-level APIs. It also ensures LDGSTS will be used under the hood, which is not guaranteed with the higher-level APIs.

> **Note**
>
> The `cooperative_groups::memcpy_async` API is less efficient than the other APIs in this example because it automatically commits each copy operation immediately upon launch, preventing the optimization of batching multiple copies before a single commit operation that the other APIs enable.
