---
title: "Grouped Rank2K Scheduler"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#grouped-rank2k-scheduler"
---

# [Grouped Rank2K Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#grouped-rank2k-scheduler)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#grouped-rank2k-scheduler "Permalink to this headline")

The previous section described the operation of the scheduler used
for grouped GEMM kernels. While this scheduler is sufficient for
correctly implementing grouped Rank2K operations (i.e., SYR2K and HER2K), it leads to significant inefficiencies.

We next describe these inefficiencies as well as how the CUTLASS
grouped Rank2K scheduler overcomes them.
