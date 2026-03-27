---
title: "Tiled MMA and Copy"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#tiled-mma-and-copy"
---

## [Tiled MMA and Copy](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tiled-mma-and-copy)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tiled-mma-and-copy "Permalink to this headline")

The Tiled MMA or Copy are tilings of MMA atoms resp. Copy atoms
across threads and data, with possible permutations applied to the
resulting tiling. This layer is most analogous to the warp level
tiling of MMA instructions in CUTLASS 2.x. However, it views the tiling
from the perspective of all threads participating in the operation
and generalizes the concept to copy operations as well. The purpose
of this layer is to build composable GPU micro-kernels out of a plethora
of hardware accelerated math and data movement operations, each with their
unit layouts in threads and data. The tiled MMA and Copy types present
all these various hardware accelerated CuTe Atoms with a single, consistent
API.

The resulting tiled operation acts as a single MMA or copy operation
that users can invoke in the “inner” loop
of the three-nested-loops pseudocode
at the top of this document using `cute::gemm()` or `cute::copy()`.

We call this API “tiled” because it constructs
larger operations out of the Atoms provided by CuTe,
as if fitting together individual tiles
to build a reusable component of a mosaic.
For example, CuTe might provide an MMA Atom
that users can call on a single warp,
for fixed M, N, and K dimensions.
CUTLASS can then use CuTe operations like `make_tiled_mma`
to turn this Atom into an operation
that works on an entire thread block,
for larger M, N, and K dimensions.
