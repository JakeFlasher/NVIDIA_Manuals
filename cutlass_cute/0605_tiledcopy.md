---
title: "TiledCopy"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#tiledcopy"
---

### [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tiledcopy)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tiledcopy "Permalink to this headline")

First, we can replace the `tA` partitioning and `tB` partitioning with `TiledCopy` partitioning, which provides for more complex partitioning patterns and checked dispatch to specific copy instructions.

As a first example, lets look at the `TiledCopy` that `gemm_nt` generates.

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},  // Atom: Copy TAs as if they were uint128_t
                                    Layout<Shape<_32,_8>>{},                    // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});                   // Val layout  4x1 m-major
  print_latex(copyA);
```

The easiest way to see what this `TiledCopy` does is to look at the partition pattern in LaTeX.
![TiledCopyA.png](images/__________1.png)
On the left is the source-tensor partitioning and on the right is the destination-tensor partitioning. The partition patterns are the same for this case, but there exist PTX instructions which require different patterns in the source and destination. The diagram shows that each thread reads 4x1 `TA` elements and there are 32x8 threads. The `UniversalCopy<uint128_t>` forces the instruction to use a 128-bit copy instruction. If the partition (of `sA` or `gA` in this case) does not result in 4 `TA` elements that cannot be vectorized to a 128-bit load/store, then CuTe will statically fail with an error message to that effect.

To use the `TiledCopy`, the kernel writes

```cpp
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);            // (CPY,CPY_M,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);              // (CPY,CPY_M,CPY_K)
```

which applies the source-tensor partitioning to `gA` via `partition_S` and applies the destination-tensor partitioning to `sA` via `partition_D`. The first mode, `CPY`, of the result tensors hold all of the elements that a single instruction will consume. In this case, that mode should have size-4 since there are four `TA=float` elements in a single 128-bit `uint128_t`.

Once the partition has been performed, we can execute the `copy` on the thread-partitioned tensors using the provided instruction in `copy_a`.

```cpp
cute::copy(copy_a, tAgA, tArA);
```
