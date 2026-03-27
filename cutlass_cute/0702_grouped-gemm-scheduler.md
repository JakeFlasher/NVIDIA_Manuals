---
title: "Grouped GEMM Scheduler"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#grouped-gemm-scheduler"
---

# [Grouped GEMM Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#grouped-gemm-scheduler)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#grouped-gemm-scheduler "Permalink to this headline")

The scheduler used by grouped GEMM assigns tiles in the group to threadblocks in a round-robin
fashion.

Consider, for example, the threadblock-to-tile mapping that occurs for a group of four GEMMs
each consisting of a grid of 2x2 tiles. Suppose that eight threadblocks are launched. The
figure below illustrates the threadblock ID assigned to each tile in each GEMM in the group.

![ALT](images/_______-____-__________1.png)

A similar mapping for problems that do not have the same number of tiles
is shown below:

![ALT](images/_______-____-__________2.png)
