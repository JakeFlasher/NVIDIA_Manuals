---
title: "4.3.2.2. Freeing Memory"
section: "4.3.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#freeing-memory"
---

### [4.3.2.2. Freeing Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#freeing-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#freeing-memory "Permalink to this headline")

`cudaFreeAsync()` asynchronously frees device memory in a stream-ordered
fashion, meaning the memory deallocation is assigned to a specific CUDA stream
and does not block the host or other streams.

The user must guarantee that the free operation happens after the allocation
operation and any uses of the allocation. Any use of the allocation
after the free operation starts results in undefined behavior.

Events and/or stream synchronizing operations should be used to guarantee any
access to the allocation from other streams is complete before the free operation
begins, as illustrated in the following example.

```c++
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
//stream2 must wait for the allocation to be ready before accessing
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 must wait for stream2 to finish accessing the allocation before
// freeing the allocation
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

Memory allocated with `cudaMalloc()` can be freed with with
`cudaFreeAsync()`. As above, all accesses to the memory must be
complete before the free operation begins.

```c++
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```

Likewise, memory allocated with `cudaMallocAsync` can be freed with
`cudaFree()`. When freeing such allocations through the `cudaFree()` API,
the driver assumes that all accesses to the allocation are complete and
performs no further synchronization. The user can use `cudaStreamQuery` /
`cudaStreamSynchronize` / `cudaEventQuery` / `cudaEventSynchronize` /
`cudaDeviceSynchronize` to guarantee that the appropriate asynchronous work
is complete and that the GPU will not try to access the allocation.

```c++
cudaMallocAsync(&ptr, size,stream);
kernel<<<..., stream>>>(ptr, ...);
// synchronize is needed to avoid prematurely freeing the memory
cudaStreamSynchronize(stream);
cudaFree(ptr);
```
