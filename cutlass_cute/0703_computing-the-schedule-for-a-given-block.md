---
title: "Computing the schedule for a given block"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#computing-the-schedule-for-a-given-block"
---

## [Computing the schedule for a given block](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#computing-the-schedule-for-a-given-block)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#computing-the-schedule-for-a-given-block "Permalink to this headline")

Each threadblock in the grouped GEMM computes its own schedule by calling
the `next_tile()` method described above.

To do this, the threadblock’s `ProblemVisitor` maintains a `thread_idx`
member that is initialized to `blockIdx.x` and is incremented by
`gridDim.x` between each tile computed (only the x dimension is used)
in the launch configuration for grouped kernels). The scheduler must
then figure out which GEMM in the group `tile_idx` belongs to, and which tile
within that problem it maps to.

1. **Determining which GEMM `tile_idx` maps to:** The scheduler determines
the GEMM to which `tile_idx` belongs by iterating through GEMMs starting with
the most-recently visited GEMM, and adding the number of tiles within that
GEMM to a running variable `problem_tile_start`. The scheduler has found the
correct problem for this tile when `problem_tile_start <= tile_idx < problem_tile_start + tiles_in_problem`.
2. **Determining the tile within a GEMM `tile_idx` maps to:** Once the GEMM
to which `tile_idx` maps has been located, the specific tile within that
GEMM that this block should compute is given by `tile_idx - problem_tile_start`.
Simple rasterization is then performed to map this one-dimensional tile ID
into the two-dimensional coordinate of the tile to compute in the GEMM.

We describe how this search is accelerated in [Scheduler Modes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#scheduler-modes).
