---
title: "Precomputing schedules on the host: GroupScheduleMode::kHostPrecompute"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#precomputing-schedules-on-the-host-groupschedulemode-khostprecompute"
---

## [Precomputing schedules on the host: GroupScheduleMode::kHostPrecompute](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#precomputing-schedules-on-the-host-groupschedulemode-khostprecompute)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#precomputing-schedules-on-the-host-groupschedulemode-khostprecompute "Permalink to this headline")

This scheduler attempts to reduce the amount of scheduling performed on the device
by precomputing on the host the sequence of problems that will
be accessed by each block. As described above, all that is needed to map tile_idx  to
the specific tile within a problem to compute is the problem ID and the problem’s
starting tile (among all of the tiles in the group). Thus, this scheduler precomputes
the problem index and problem starting tile for each tile computed by each block.

The schedule for an individual block is represented as an array of
`(problem_idx, problem_starting_tile)` tuples. There is one such array per block.
These arrays are produced on the host and copied over to the device. This
representation is optimized for the case in which blocks compute at most one
tile per problem. When a block computes multiple tiles per problem in the group,
the representation above will result in duplicate entries, and thus will be
suboptimal (e.g., `[(3, 20), (3, 20)]` for a block that computes two tiles in
problem 3, which has starting tile index 20).
We have chosen to use the representation described above because grouped kernels
themselves are typically most beneficial when problem sizes are small, and, thus,
blocks compute at most one tile per problem.
