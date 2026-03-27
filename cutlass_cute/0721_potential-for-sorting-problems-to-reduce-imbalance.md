---
title: "Potential for sorting problems to reduce imbalance"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#potential-for-sorting-problems-to-reduce-imbalance"
---

## [Potential for sorting problems to reduce imbalance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#potential-for-sorting-problems-to-reduce-imbalance)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#potential-for-sorting-problems-to-reduce-imbalance "Permalink to this headline")

A simple way to potentially reduce load imbalance is to sort the problems in a group in
descending order of their K dimension. This can help to improve load balance
because tiles in a group are assigned in a round-robin fashion to blocks
sequentially, so every block will always be assigned next the tile with
the highest K dimension available.

Considering the example described above, sorting the problem sizes before
executing grouped GEMM improves the runtime of this grouped GEMM on GA100 with each
scheduling mode by around 30%.

To ease the process of sorting groups and their associated metadata in this
manner, the device-level grouped kernels provide a `sort_problems()` method.
An example of how to use this may be found in the [grouped GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped/gemm_grouped.cu).

Finally, while sorting problems can be helpful in certain scenarios, it is
not guaranteed to improve performance. In some cases, performance can
decrease when sorting problems due to additional conflicting factors that
affect GEMM performance. We recommend profiling your grouped kernel with
and without sorting to see whether it helps in your case.
