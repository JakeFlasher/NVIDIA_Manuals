---
title: "4.13.6. Manage Utilization of L2 set-aside cache"
section: "4.13.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#manage-utilization-of-l2-set-aside-cache"
---

## [4.13.6. Manage Utilization of L2 set-aside cache](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#manage-utilization-of-l2-set-aside-cache)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#manage-utilization-of-l2-set-aside-cache "Permalink to this headline")

Multiple CUDA kernels executing concurrently in different CUDA streams may have a different access policy window assigned to their streams. However, the L2 set-aside cache portion is shared among all these concurrent CUDA kernels. As a result, the net utilization of this set-aside cache portion is the sum of all the concurrent kernels’ individual use. The benefits of designating memory accesses as persisting diminish as the volume of persisting accesses exceeds the set-aside L2 cache capacity.

To manage utilization of the set-aside L2 cache portion, an application must consider the following:

- Size of L2 set-aside cache.
- CUDA kernels that may concurrently execute.
- The access policy window for all the CUDA kernels that may concurrently execute.
- When and how L2 reset is required to allow normal or streaming accesses to utilize the previously set-aside L2 cache with equal priority.
