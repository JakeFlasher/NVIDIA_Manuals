---
title: "Volta"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#volta"
---

## [Volta](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#volta)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#volta "Permalink to this headline")

This and the following sections show examples of how to construct MMA atoms.
We don’t try to explain this for all GPU architectures and MMAs.
Instead, we use selected examples to illustrate the process
of developing new atoms.

Volta architecture implements an HMMA instruction where a group of 8 threads called a quadpair (QP) collaborate to share data and perform an 8x8x4 (fp32 or fp16) matrix multiply-accumulate. (since a warp is 32 threads wide, it would perform an MMA across 4 QPs for a tile size of 16x16x4).

We first take a look at how we would take the ISA semantics of thread and data partitioning for the HMMA instruction, and encode it in a Traits struct. The HMMA NT instruction has the thread-data layout:

![HMMA.8x8x4.NT.png](images/______1.png)
