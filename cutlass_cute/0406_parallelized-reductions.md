---
title: "Parallelized Reductions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions"
---

### [Parallelized Reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#parallelized-reductions)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#parallelized-reductions "Permalink to this headline")

**Split K - reduction across threadblocks**

Matrix product computations expose parallelism among _O(MN)_ independent inner product
computations. For sufficiently large problem sizes, a GEMM kernel in CUTLASS may approach
the theoretical maximum computational throughput. For small problems, however, there are
too few threadblocks to efficiently occupy the entire GPU.

As a recourse, parallelizing the reduction performed during the inner product computation
enables more threadblocks to execute concurrently while still taking advantage of the throughput
benefits of large threadblock-level GEMM tiles.

CUTLASS implements parallel reductions across threadblocks by partitioning the GEMM _K_ dimension
and launching an additional set of threadblocks for each partition. Consequently, we refer to
this strategy within CUTLASS as “parallel reduction splitK.” The “parallel reduction splitK” strategy
requires the execution of 2 kernels: partitionedK GEMM, and batched reduction.

PartitionedK GEMM resembles one flavor of batched strided GEMM. Instead of requiring users
to specify the problem size of each batch, partitionedK GEMM asks for the overall problem size and the
number of partitions that will be applied along the K dimension for operands A and B. For example,
parameters of m=128, n=128, k=4096 and partition=16 will result in 16 batched strided GEMMs
with each batch of m=128, n=128, k=256. PartitionedK also allows scenario where k is not divisible
by the partition count.

For example, parameters of m=128, n=128, k=4096 and partition=20
will result in 20 batched strided GEMMs.
The first 19 batches will have m=128, n=128, and k=4096/20=204,
and the last batch will have m=128, n=128, and k=220.

The batched reduction kernel takes as input the output (C) of partitionedK GEMM,
and performs a reduction along the K-dimension.
Users must manage workspace memory to store this intermediate result.

**Sliced K - reduction across warps**

Similar to the split-k scenario, sliced-k aims at improving the efficiency of kernels
with smaller M and N dimensions, but large K dimension.
At the thread-block level, the parameters CtaTileN and CtaTileM expose parallelism
by partitioning the work among warps.
Larger warpTiles expose better instruction-level parallelism (ILP) and reuse,
but also limit the number of warps running per threadblock, which reduces efficiency.

In order to improve efficiency in such scenarios, partitioning the warpTiles also along ctaTileK
helps use the hardware more efficiently by allowing more warps to run concurrently in a CTA.
Sliced-k kernels break down a threadblock’s computation among participating warps
not just among the CtaTileN, CtaTileM dimension, but also the CtaTileK dimension.
Thus, sliced-k entails a small cost in form of a reduction
which has to happen at the end among the participating warps.
This is because each warp computes using only a “slice” of CtaTileK,
so each warp only has a partial sum before the reduction.
