---
title: "Potential problems with imbalanced K dimension"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#potential-problems-with-imbalanced-k-dimension"
---

## [Potential problems with imbalanced K dimension](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#potential-problems-with-imbalanced-k-dimension)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#potential-problems-with-imbalanced-k-dimension "Permalink to this headline")

To ensure that compute load is balanced evenly across blocks, it is important
that the sum of the K dimensions among all tiles a block computes be similar
to that of other blocks; if one block computes far more tiles with a large
value of K than other blocks, it may take longer than the other blocks.

For example, consider the following group of GEMMs:

```console
0 1152x768x128
1 1152x768x1024
2 768x1152x128
3 768x1152x1024
```

If a tile size of 128x128 is used, then each problem will have 54 tiles.
Thus, there are 216 tiles across the group.

Suppose this grouped GEMM is run on GA100, which has 108 SMs. Suppose that
the occupancy given the parameters of the grouped GEMM is one – one threadblock
can be active at a time on an SM. The grouped GEMM will, thus, run with 108
persistent threadblocks, each of which computes (256 / 108) = 2 tiles.

Under the round-robin assignment of tiles to threadblocks employed by
the grouped GEMM scheduler, the assignment of tiles to threadblocks
in this GEMM will be as follows:

```console
Threadblocks 0-53:     Tiles of size 128x128x128  from problem 0
Threadblocks 54-107:   Tiles of size 128x128x1024 from problem 1
Threadblocks 0-53:     Tiles of size 128x128x128  from problem 2
Threadblocks 54-107:   Tiles of size 128x128x1024 from problem 3
```

Following this assignment, threadblocks 54-107 perform significantly more
work than threadblocks 0-53 because they compute two tiles with a K
dimension of 1024, whereas threadblocks 0-53 compute two tiles with K
dimension of only 128.

Due to this imbalanced assignment, threadblocks 54-107 will run
significantly longer than threadblocks 0-53, leaving threadblocks
0-53 idle for a large fraction of time.

Clearly, a better assignment of tiles to threadblocks for this
example would involve all threadblocks computing one tile with
a K dimension of 1024 and one tile with a K dimension of 128.
This would better balance the workload among threadblocks.
