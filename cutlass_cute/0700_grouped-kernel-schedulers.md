---
title: "Grouped Kernel Schedulers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#grouped-kernel-schedulers"
---

# [Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#grouped-kernel-schedulers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#grouped-kernel-schedulers "Permalink to this headline")

CUTLASS’s grouped kernel is a persistent kernel which launches multiple problems (e.g., GEMMs, SYR2Ks) within a
single CUDA kernel launch.

Unlike a conventional GEMMs in CUTLASS, which launch a number of threadblocks equal to the number
of tiles in the GEMM, CUTLASS grouped kernels typically launch a number of threadblocks that is
fewer than the total number of tiles across all problems in the group. Each threadblock is then
responsible for computing one or more tiles among the problems in the group. The grouped kernel
_scheduler_ (referred to as the _problem visitor_ in code) is responsible for assigning each
threadblock the sequence of tiles that it will compute within the group.

This document provides background on the functionality of the grouped kernel scheduler, and describes
various optimizations to the grouped kernel scheduler.

**Outline**

- [Introduction to Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#introduction-to-grouped-kernel-schedulers)
- [Grouped GEMM Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#grouped-gemm-scheduler)
- [Grouped Rank2K Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#grouped-rank2k-scheduler)
- [Scheduler Modes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#scheduler-modes)
- [Improving Load Balance by Sorting Problems](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#improving-load-balance-by-sorting-problems)
