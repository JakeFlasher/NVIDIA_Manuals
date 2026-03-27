---
title: "4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic"
section: "4.3.4.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudamempoolreuseallowopportunistic"
---

#### [4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreuseallowopportunistic)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreuseallowopportunistic "Permalink to this headline")

When the `cudaMemPoolReuseAllowOpportunistic` policy is enabled, the allocator
examines freed allocations to see if the free operations stream order semantic has been
met, for example the stream has passed the point of execution indicated by the
free operation. When this policy is disabled, the allocator will still reuse memory made
available when a stream is synchronized with the CPU. Disabling this policy
does not stop the `cudaMemPoolReuseFollowEventDependencies` from applying.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// after some time, the kernel finishes running
wait(10);

// When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
// can be fulfilled with the prior allocation based on the progress of originalStream.
cudaMallocAsync(&ptr2, size, otherStream);
```
