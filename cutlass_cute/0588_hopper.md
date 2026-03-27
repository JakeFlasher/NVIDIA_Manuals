---
title: "Hopper"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#hopper"
---

## [Hopper](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#hopper)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#hopper "Permalink to this headline")

Now, we are ready to take a look at the much larger GMMA operation (Group MMA) first introduced with Hopper architecture. These MMA instructions operate at the granularity of 128 threads (4 warps), which are collectively referred to as a warpgroup.
