---
title: "CUTLASS GEMM Model"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#cutlass-gemm-model"
---

## [CUTLASS GEMM Model](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-gemm-model)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-gemm-model "Permalink to this headline")

CUTLASS implements algorithms that express
the classical “triply nested loop” GEMM algorithm
with a tiled structure mirroring the above hierarchy.

The following pseudocode describes the model for a GEMM kernel
targeting a warp-synchronous matrix multiply instruction like `mma.sync.`
The entire operation is referred to as “Gemm,”
as it is assumed that an epilogue operation
performs the general matrix update similar to BLAS.
This is pseudocode and is only meant to illustrate which parts of the layers
correspond to the inner or outer loops of the GEMM.

```c++
// cutlass::gemm::kernel::GemmUniversal: ClusterTileM and ClusterTileN loops
//   are either rasterized by the hardware or scheduled by the kernel in persistent kernels.
// Parallelism over thread block clusters
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
    // No loop unrolling is performed at this stage
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
      // TiledMma uses the hardware instruction provided through its Mma_Atom
      // TiledMma's atom layout, value layout, and permutations define the iteration order
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

            // TiledMma's vector mode dispatches to the underlying instruction.
            mma.call(d, a, b, c);
          } // tiled_mma_n
        } // tiled_mma_m
      } // tiled_mma_k
    } // k_tile mainloop
  } // cluster_m
} // cluster_n
```

The first three nested `for` loops
correspond to parallelism over thread block clusters.
The code does not actually express them as explicit `for` loops.
Instead, the parallelization scheme over tiles
is implied by CUDA grid launch semantics.
However, for persistent kernels,
these three loops are expressed in the source code
as a single `while` loop that queries the
[work tile scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp)
for problem tiles on which to compute.

Inside the three nested `for` loops,
one finds code that pulls matrix tiles
from global memory into more “local” memory
(like shared memory or registers)
and computes MMAs.
These tiled copy and tiled mma iterations are generally
fully static and get fully unrolled.
