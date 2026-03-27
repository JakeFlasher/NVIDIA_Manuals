---
title: "2.2.3.6. Caches"
section: "2.2.3.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels--caches"
---

### [2.2.3.6. Caches](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#caches)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#caches "Permalink to this headline")

GPU devices have a multi-level cache structure which includes L2 and L1 caches.

The L2 cache is located on the device and is shared among all the SMs.  The size of the L2 cache can be queried with the `l2CacheSize` device property element from the function `cudaGetDeviceProperties`.

As described above in [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory), L1 cache is physically located on each SM and is the same physical space used by shared memory.  If no shared memory is utilized by a kernel, the entire physical space will be utilized by the L1 cache.

The L2 and L1 caches can be controlled via functions that allow the developer to specify various caching behaviors.  The details of these functions are found in  [Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernel-l1-shared-config), [L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#advanced-kernels-l2-control), and [Low-Level Load and Store Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#low-level-load-store-functions).

If these hints are not used, the compiler and runtime will do their best to utilize the caches efficiently.
