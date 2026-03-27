---
title: "2.2.5. Atomics"
section: "2.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#atomics"
---

## [2.2.5. Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#atomics)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#atomics "Permalink to this headline")

Performant CUDA kernels rely on expressing as much algorithmic parallelism as possible.  The asynchronous nature of GPU kernel execution requires that threads operate as independently as possible.  It’s not always possible to have complete independence of threads and as we saw in [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory), there exists a mechanism for threads in the same thread block to exchange data and synchronize.

On the level of an entire grid there is no such mechanism to synchronize all threads in a grid.  There is however a mechanism to provide synchronous access to global memory locations via the use of atomic functions.  Atomic functions allow a thread to obtain a lock on a global memory location and perform a read-modify-write operation on that location.  No other thread can access the same location while the lock is held.  CUDA provides atomics with the same behavior as the C++ standard library atomics as `cuda::std::atomic` and `cuda::std::atomic_ref`.  CUDA also provides extended C++ atomics `cuda::atomic` and `cuda::atomic_ref` which allow the user to specify the [thread scope](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-thread-scopes) of the atomic operation.  The details of atomic functions are covered in [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions).

An example usage of `cuda::atomic_ref` to perform a device-wide atomic addition is as follows, where `array` is an array of floats, and `result` is a float pointer to a location in global memory which is the location where the sum of the array will be stored.

```c++
__global__ void sumReduction(int n, float *array, float *result) {
   ...
   tid = threadIdx.x + blockIdx.x * blockDim.x;

   cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);
   result_ref.fetch_add(array[tid]);
   ...
}
```

Atomic functions should be used sparingly as they enforce thread synchronization that can impact performance.
