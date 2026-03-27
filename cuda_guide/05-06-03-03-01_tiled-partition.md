---
title: "5.6.3.3.1. tiled_partition"
section: "5.6.3.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#tiled-partition"
---

#### [5.6.3.3.1. tiled_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#tiled-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#tiled-partition "Permalink to this headline")

```c++
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);
```

```c++
thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
```

The `tiled_partition` method is a collective operation that partitions the parent group into a one-dimensional, row-major, tiling of subgroups. A total of ((size(parent)/tilesz) subgroups will be created, therefore the parent group size must be evenly divisible by the `Size`. The allowed parent groups are `thread_block` or `thread_block_tile`.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution. Functionality is limited to native hardware sizes, 1/2/4/8/16/32 and the `cg::size(parent)` must be greater than the `Size` parameter. The templated version of `tiled_partition` supports 64/128/256/512 sizes as well, but some additional steps are required on Compute Capability 7.5 or lower, refer to [class thread_block_tile](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cg-api-thread-block-tile) for details.

**Codegen Requirements:** Compute Capability 5.0 minimum, C++11 for sizes larger than 32
