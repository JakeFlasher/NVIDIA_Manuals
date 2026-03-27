---
title: "4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies"
section: "4.3.4.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#cudamempoolreuseallowinternaldependencies"
---

#### [4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreuseallowinternaldependencies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreuseallowinternaldependencies "Permalink to this headline")

Failing to allocate and map more physical memory from the OS, the driver will
look for memory whose availability depends on another stream’s pending
progress. If such memory is found, the driver will insert the required
dependency into the allocating stream and reuse the memory.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// When cudaMemPoolReuseAllowInternalDependencies is enabled
// and the driver fails to allocate more physical memory, the driver may
// effectively perform a cudaStreamWaitEvent in the allocating stream
// to make sure that future work in ‘otherStream’ happens after the work
// in the original stream that would be allowed to access the original allocation.
cudaMallocAsync(&ptr2, size, otherStream);
```
