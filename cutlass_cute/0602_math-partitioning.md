---
title: "Math partitioning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#math-partitioning"
---

### [Math partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#math-partitioning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#math-partitioning "Permalink to this headline")

The kernel now has tiles of shared memory copied in from global memory. We now want to create an efficient way to compute and accumulate the matrix product on that tile of shared memory. A trivial way to do this would be to use a single thread and compute directly.

```cpp
if (thread0()) {
  for (int m = 0; m < size<0>(gC); ++m) {
    for (int n = 0; n < size<1>(gC); ++n) {
      for (int k = 0; k < size<1>(sA); ++k) {
        gC(m,n) += sA(m,k) * sB(n,k);
      }
    }
  }
}
```

This would work, but we have lots of threads to use inside this CTA, so let’s use them!

If we partition the output tile `gC` across the threads in the CTA, then each thread can compute its own subtensor. There are lots of ways this partitioning could occur, however.

The `gemm_nt` and `gemm_tn` functions define one more layout of _threads_:

```cpp
  // Define thread layouts (static)
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx; m-major
```

This is a m-major 16x16 layout of threads which will be used to partition a 128x128 tile of `C`-data, resulting in each thread computing its own 8x8 subtensor of `gC`.

Again, the conditions on the thread layouts are checked inside the kernel.

```cpp
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N
```

These thread layouts are then used to partition the tiles of data in global memory and shared memory

```cpp
  // Partition sA (M,K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (N,K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K
```

where we’ve used the same projection-style interface to avoid applying the `N`-mode of `tC` to the `(BLK_M,BLK_K)` shape of `sA` and avoid applying the `M`-mode of `tC` to the `(BLK_N,BLK_K)` shape of `sB`.

![tC_partitioning.png](images/____-_____________1.png)
This diagram shows a `tC` layout, highlights two threads in green and blue, shows the projections of the `tC` layout, and finally highlights the subtensors within `sA`, `sB`, and `gC` that `tCsA`, `tCsB`, and `tCgC` represent.

With the data partitioned across the threads, _every thread_ can now participate in the compute step by writing

```cpp
gemm(tCsA, tCsB, tCrC);
```

because every thread owns different subtensors of the data to be computed.
