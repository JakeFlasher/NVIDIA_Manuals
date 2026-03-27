---
title: "4.3.3. Memory Pools"
section: "4.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#memory-pools"
---

## [4.3.3. Memory Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-pools)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-pools "Permalink to this headline")

Memory pools encapsulate virtual address and physical memory resources that
are allocated and managed according to the pools attributes and properties.
The primary aspect of a memory pool is the kind and location of memory it
manages.

All calls to `cudaMallocAsync` use resources from memory pool. If
a memory pool is not specified, `cudaMallocAsync` uses the current
memory pool of the supplied stream’s device. The current memory pool for a
device may be set with `cudaDeviceSetMempool` and queried with
`cudaDeviceGetMempool`. Each device has a default memory pool, which is
active if `cudaDeviceSetMempool` has not been called.

The API `cudaMallocFromPoolAsync` and [c++ overloads of
cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb)
allow a user to specify the pool to be used for an allocation without setting
it as the current pool. The APIs `cudaDeviceGetDefaultMempool` and
`cudaMemPoolCreate` return handles to memory pools. `cudaMemPoolSetAttribute`
and `cudaMemPoolGetAttribute` control the attributes of memory pools.

> **Note**
>
> The mempool current to a device will be local to that device. So allocating
> without specifying a memory pool will always yield an allocation local to
> the stream’s device.
