---
title: "4.4.7. Asynchronous Data Movement"
section: "4.4.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#asynchronous-data-movement"
---

## [4.4.7. Asynchronous Data Movement](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#asynchronous-data-movement)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#asynchronous-data-movement "Permalink to this headline")

Cooperative Groups `memcpy_async` functionality in CUDA provides a way to perform asynchronous memory copies between global memory and shared memory.
`memcpy_async` is particularly useful for optimizing memory transfers and overlapping computation with data transfer to improve performance.

The `memcpy_async` function is used to start an asynchronous load from global memory to shared memory.
`memcpy_async` is intended to be used like a “prefetch” where data is loaded before it is needed.

The `wait` function forces all threads in a group to wait until the asynchronous memory transfer is completed.
`wait` must be called by all threads in the group before the data can be accessed in shared memory.

The following example shows how to use `memcpy_async` and `wait` to prefetch data.

```c++
namespace cg = cooperative_groups;

cg::thread_group my_group = cg::this_thread_block();

__shared__ int shared_data[];

// Perform an asynchronous copy from global memory to shared memory
cg::memcpy_async(my_group, shared_data + my_group.rank(), input + my_group.rank(), sizeof(int));

// Hide latency by doing work here. Cannot use shared_data

// Wait for the asynchronous copy to complete
cg::wait(my_group);

// Prefetched data is now available
```

See the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-async-header) for more information.
