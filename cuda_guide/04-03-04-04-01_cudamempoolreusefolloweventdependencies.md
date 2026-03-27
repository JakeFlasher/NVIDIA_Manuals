---
title: "4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies"
section: "4.3.4.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudamempoolreusefolloweventdependencies"
---

#### [4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreusefolloweventdependencies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreusefolloweventdependencies "Permalink to this headline")

Before allocating more physical GPU memory, the allocator examines dependency
information established by CUDA events and tries to allocate from memory freed
in another stream.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event,originalStream);

// waiting on the event that captures the free in another stream
// allows the allocator to reuse the memory to satisfy
// a new allocation request in the other stream when
// cudaMemPoolReuseFollowEventDependencies is enabled.
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);
```
