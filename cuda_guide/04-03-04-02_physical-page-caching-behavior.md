---
title: "4.3.4.2. Physical Page Caching Behavior"
section: "4.3.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#physical-page-caching-behavior"
---

### [4.3.4.2. Physical Page Caching Behavior](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#physical-page-caching-behavior)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#physical-page-caching-behavior "Permalink to this headline")

By default, the allocator tries to minimize the physical memory owned by a
pool. To minimize the OS calls to allocate and free physical memory,
applications must configure a memory footprint for each pool. Applications can
do this with the release threshold attribute
(`cudaMemPoolAttrReleaseThreshold`).

The release threshold is the amount of memory in bytes a pool should hold onto
before trying to release memory back to the OS. When more than the release
threshold bytes of memory are held by the memory pool, the allocator will try
to release memory back to the OS on the next call to stream, event or device
synchronize. Setting the release threshold to UINT64_MAX will prevent the
driver from attempting to shrink the pool after every synchronization.

```c++
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

Applications that set `cudaMemPoolAttrReleaseThreshold` high enough to
effectively disable memory pool shrinking may wish to explicitly shrink a
memory pool’s memory footprint. `cudaMemPoolTrimTo` allows applications
to do so. When trimming a memory pool’s footprint, the `minBytesToKeep`
parameter allows an application to hold onto a specified amount of memory,
for example the amount it expects to need in a subsequent phase of execution.

```c++
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// application phase needing a lot of memory from the stream-ordered allocator
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j],size[j], stream);
    }
    kernel<<<...,stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// Process does not need as much memory for the next phase.
// Synchronize so that the trim operation will know that the allocations are no
// longer in use.
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// Some other process/allocation mechanism can now use the physical memory
// released by the trimming operation.
```
