---
title: "4.4.4. Creating Cooperative Groups"
section: "4.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#creating-cooperative-groups"
---

## [4.4.4. Creating Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-cooperative-groups)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-cooperative-groups "Permalink to this headline")

Groups are created by partitioning a parent group into subgroups.
When a group is partitioned, a group handle is created to manage the resulting subgroup.
The following partitioning operations are available to developers:

| Partition Type | Description |
| --- | --- |
| tiled_partition | Divides parent group into a series of fixed-size subgroups arranged in a one-dimensional, row-major format. |
| stride_partition | Divides parent group into equally-sized subgroups where threads are assigned to subgroups in a round-robin manner. |
| labeled_partition | Divides parent group into one-dimensional subgroups based on a conditional label, which can be any integral type. |
| binary_partition | Specialized form of labeled partitioning where label can only be “0” or “1”. |

The following example shows how a tiled partition is created:

```c++
namespace cg = cooperative_groups;
// Obtain the current thread's cooperative group
cg::thread_block my_group = cg::this_thread_block();

// Partition the cooperative group into tiles of size 8
cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(cta);

// do work as my_subgroup
```

The best partitioning strategy to use depends on the context. More information is available in the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-partition-header).
