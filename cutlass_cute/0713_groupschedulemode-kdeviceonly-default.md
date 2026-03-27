---
title: "GroupScheduleMode::kDeviceOnly (default)"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#groupschedulemode-kdeviceonly-default"
---

## [GroupScheduleMode::kDeviceOnly (default)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#groupschedulemode-kdeviceonly-default)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#groupschedulemode-kdeviceonly-default "Permalink to this headline")

This scheduler mode performs all scheduling work on the device. It parallelizes
the search for the problem that `tile_idx` maps to by having each thread “own”
a different problem and determine whether `tile_idx` falls within the range of
that problem.

`GroupScheduleMode::kDeviceOnly` performs this parallelization in a warp-wide
fashion. Each thread in the warp loads a problem size indexed by its lane id and
computes the number of tiles in that problem. A warp-wide prefix sum is used to find
the starting tiles for the set of problems the warp is looking at. At the end of the
prefix sum, each thread holds the starting tile index and tile count for a unique
problem in the group.

While `tile_idx` remains within the range of the problems currently hosted by the
warp, each thread will check whether `tile_idx`  is in the range of its current
problem. The matching problem index and its starting tile are then broadcasted to all
threads in the warp.
