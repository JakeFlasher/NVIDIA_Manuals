---
title: "CUTLASS’s abstractions for Hopper features"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#cutlass-s-abstractions-for-hopper-features"
---

## [CUTLASS’s abstractions for Hopper features](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-s-abstractions-for-hopper-features)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-s-abstractions-for-hopper-features "Permalink to this headline")

CUTLASS now includes abstractions
for the following features introduced in Hopper.

1. Thread block cluster - level synchronization and query
[APIs](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/cluster_sm90.hpp)
2. Abstractions for new
[barrier instructions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/barrier.h)
which help with efficient synchronization
of threads within a thread block cluster.
