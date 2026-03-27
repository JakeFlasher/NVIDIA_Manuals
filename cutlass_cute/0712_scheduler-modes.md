---
title: "Scheduler modes"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#scheduler-modes"
---

# [Scheduler modes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#scheduler-modes)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#scheduler-modes "Permalink to this headline")

The grouped kernel schedulers come with two different modes for finding
the next tile for a block to compute. These techniques are controlled by
the [`cutlass::gemm::kernel::GroupScheduleMode`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/grouped_problem_visitor.h) enum.
We describe each mode in greater detail below.
