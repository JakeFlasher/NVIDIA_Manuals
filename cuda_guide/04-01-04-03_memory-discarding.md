---
title: "4.1.4.3. Memory Discarding"
section: "4.1.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#memory-discarding"
---

### [4.1.4.3. Memory Discarding](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-discarding)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-discarding "Permalink to this headline")

The `cudaMemDiscardBatchAsync` API allows applications to inform the CUDA runtime that the contents
of specified memory ranges are no longer useful. The Unified Memory driver performs automatic memory
transfers due to fault-based migration or memory evictions to support device memory oversubscription.
These automatic memory transfers can sometimes be redundant, which severely decreases performance.
Marking an address range as ‘discard’ will inform the Unified Memory driver that the application has
consumed the contents in the range and there is no need to migrate this data on prefetches or page evictions
in order to make room for other allocations. Reading a discarded page without a subsequent write access
or prefetch will yield an indeterminate value. Whereas any new writes after the discard operation is guaranteed
to be seen by a subsequent read access. Concurrent accesses or prefetches to address ranges being discarded
will result in undefined behavior.

```c++
cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                    size_t *sizes,
                                    size_t count,
                                    unsigned long long flags,
                                    cudaStream_t stream);
```

The function performs a batch of memory discards on address ranges specified in `dptrs` and `sizes` arrays.
Both arrays must be of the same length as specified by `count`. Each memory range must refer to
managed memory allocated via `cudaMallocManaged` or declared via `__managed__` variables.

The `cudaMemDiscardAndPrefetchBatchAsync` API combines both discard and prefetch operations. Calling
`cudaMemDiscardAndPrefetchBatchAsync` is semantically equivalent to calling `cudaMemDiscardBatchAsync`
followed by `cudaMemPrefetchBatchAsync`, but is more optimal. This is useful when the application needs
the memory to be on the target location but does not need the contents of the memory.

```c++
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                               size_t *sizes,
                                               size_t count,
                                               struct cudaMemLocation *prefetchLocs,
                                               size_t *prefetchLocIdxs,
                                               size_t numPrefetchLocs,
                                               unsigned long long flags,
                                               cudaStream_t stream);
```

The `prefetchLocs` array specifies the destinations for prefetching, while `prefetchLocIdxs`
indicates which operations each prefetch location applies to. For example, if a batch has 10 operations
and the first 6 should be prefetched to one location while the remaining 4 to another, then
`numPrefetchLocs` would be 2, `prefetchLocIdxs` would be {0, 6}, and `prefetchLocs` would
contain the two destination locations.

**Important considerations:**

- Reading from a discarded range without a subsequent write or prefetch will return an indeterminate value
- The discard operation can be undone by writing to the range or prefetching it via `cudaMemPrefetchAsync`
- Any reads, writes, or prefetches that occur simultaneously with the discard operation result in undefined behavior
- All devices must have a non-zero value for `cudaDevAttrConcurrentManagedAccess`
