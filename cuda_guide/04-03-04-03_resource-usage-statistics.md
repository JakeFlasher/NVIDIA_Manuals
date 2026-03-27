---
title: "4.3.4.3. Resource Usage Statistics"
section: "4.3.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#resource-usage-statistics"
---

### [4.3.4.3. Resource Usage Statistics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#resource-usage-statistics)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#resource-usage-statistics "Permalink to this headline")

Querying the `cudaMemPoolAttrReservedMemCurrent` attribute of a pool reports
the current total physical GPU memory consumed by the pool. Querying the
`cudaMemPoolAttrUsedMemCurrent` of a pool returns the total size of all of
the memory allocated from the pool and not available for reuse.

The`cudaMemPoolAttr*MemHigh` attributes are watermarks recording the max
value achieved by the respective `cudaMemPoolAttr*MemCurrent` attribute
since last reset. They can be reset to the current value by using the
`cudaMemPoolSetAttribute` API.

```c++
// sample helper functions for getting the usage statistics in bulk
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}

// resetting the watermarks will make them take on the current value.
void resetStatistics(cudaMemoryPool_t memPool)
{
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```
