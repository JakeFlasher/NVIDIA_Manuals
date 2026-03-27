---
title: "5.6.3.3.2. labeled_partition"
section: "5.6.3.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#labeled-partition"
---

#### [5.6.3.3.2. labeled_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#labeled-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#labeled-partition "Permalink to this headline")

```c++
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);
```

```c++
template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```

The `labeled_partition` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a condition label and assign threads that have the same value for label into the same group.

`Label` can be any integral type.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Note:** This functionality is still being evaluated and may slightly change in the future.

**Codegen Requirements:** Compute Capability 7.0 minimum, C++11
