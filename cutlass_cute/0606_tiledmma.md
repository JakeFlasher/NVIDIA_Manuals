---
title: "TiledMMA"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#tiledmma"
---

### [TiledMMA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tiledmma)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tiledmma "Permalink to this headline")

Next, we can replace the `tC` partitioning with `TiledMMA` partitioning, which provides for more complex partitioning patterns and checked dispatch to specific MMA instructions.

As a first example, lets look at the `TiledMMA` that `gemm_nt` generates.

```cpp
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 UniversalFMA
  print_latex(mmaC);
```

The easiest way to see what this `TiledMMA` does is to look at the partition pattern in LaTeX.
![TiledMmaC.png](images/_________1.png)
On the left is the A-tensor partitioning, on the top is the B-tensor partitioning, and in the middle is the C-tensor partitioning.Because the `UniversalFMA` is a 1x1x1 MMA instruction, a 16x16x1 tiling of them results in a 16x16x1 `TiledMMA`. Other MMA instructions will have different threads involved and have different instruction sizes. In this case, all threads will read a single element from `A`, `B`, and `C` each.

To use the `TiledMMA`, the kernel writes

```cpp
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
```

which applies the A-tensor partitioning to `sA` via `partition_A`, applies the B-tensor partitioning to `sB` via `partition_B`, and applies the C-tensor partitioning to `gC` via `partition_C`. The first mode, `MMA`, of the result tensors hold all of the elements that a single instruction will consume. In this case, that mode should have size-1 since `UniversalFMA` is a 1x1x1 MMA, but in general the size of the first mode can vary and not even be the same across `tCsA`, `tCsB`, and `tCgC` depending on the MMA.

Once the partition has been performed, we can execute the `gemm` on the thread-partitioned tensors using the provided instruction in `mma`.

```cpp
cute::gemm(mma, tCsA, tCsB, tCrC);
```
