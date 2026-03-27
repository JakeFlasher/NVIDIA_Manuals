---
title: "How compute-intensive are the problems in my group?"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#how-compute-intensive-are-the-problems-in-my-group"
---

### [How compute-intensive are the problems in my group?](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#how-compute-intensive-are-the-problems-in-my-group)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#how-compute-intensive-are-the-problems-in-my-group "Permalink to this headline")

The differences in performance between `kHostPrecompute` and `kDeviceOnly` are most
noticeable for grouped kernels with low computational intensity, for which time spent in
the scheduler accounts for a significant fraction of the grouped kernel’s runtime.
Intuitively, as problems in a group decrease in computational intensity, a smaller
fraction of the overall runtime will be consumed in performing MMA operations, leading
to a larger fraction of the overall runtime being consumed by scheduling logic.

Since the scheduling modes affect only the scheduling logic of the grouped kernels,
one expects to see most benefit from `kHostPrecompute` for less computationally-intense
groups.
