---
title: "Collective API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#collective-api"
---

### [Collective API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#collective-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-api "Permalink to this headline")

A Collective is “the largest collection of threads
onto which mma atoms and copy atoms are tiled.”
That is, it is the largest number of threads in a grid
that can cooperate by leveraging hardware features
for accelerated communication and synchronization.
These hardware features include

- asynchronous array copy
(e.g., from global memory to shared memory);
- MMA instructions
for small tiles that live in shared memory;
- synchronization operations for clusters,
thread blocks, and/or warps; and/or
- hardware acceleration (such as barriers)
for ensuring that data dependencies
between asynchronous operations are met.

A Collective uses the `TiledMma` and `TiledCopy` API (see below)
to access operations that copy and perform MMA on tiles.

Different units of parallelism
(e.g., threads, warps, or thread blocks)
in a Collective might have different roles.
For example, in “warp-specialized” algorithms,
some warps may be responsible for copying data,
while others may be responsible for computation.
Nevertheless, the different units of parallelism
still need to share data and coordinate access
to the shared data. For example,
the producer warps in a warp-specialized algorithm
that copy input matrix tiles into shared memory
need to let the consumer MMA warp(s) know
that their MMA inputs are ready.
We contrast this with the `kernel::` layer API,
which schedules the collectives over _independent_ tiles in the grid.

The Collective API includes both the “mainloop”
of matrix multiply-accumulate, and the epilogue.
This API is the composition point for optimizations
such as mainloop fusions and epilogue fusions.
It is responsible for implementing
the `k_tile` loop in the above triply nested loop pseudocode.
