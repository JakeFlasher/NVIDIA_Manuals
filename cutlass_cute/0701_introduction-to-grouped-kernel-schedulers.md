---
title: "Introduction to Grouped Kernel Schedulers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#introduction-to-grouped-kernel-schedulers"
---

# [Introduction to Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#introduction-to-grouped-kernel-schedulers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#introduction-to-grouped-kernel-schedulers "Permalink to this headline")

Given a group of problem sizes and a grid of threadblocks, the scheduler’s job is to assign
tiles from problems in the group to threadblocks. Threadblocks in a grouped kernel persistently
execute a loop of querying the scheduler for the next tile to compute and performing the
kernel-level operations for that tile (e.g., MMA and epilogue). In pseudocode, this looks as
follows:

```c++
ProblemVisitor problem_visitor;

while (problem_visitor.next_tile()) {
    //
    // Get next tile index from scheduler
    //

    //
    // Compute MMA and epilogue
    //

    // Inform the scheduler that we are done with the current tile
    problem_visitor.advance(gridDim.x);
}
```

The key functionality of the grouped kernel scheduler lies in the `next_tile()` method,
which determines which tile in the group the calling threadblock should compute next, if any.
