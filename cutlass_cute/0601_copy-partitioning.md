---
title: "Copy partitioning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#copy-partitioning"
---

### [Copy partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#copy-partitioning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#copy-partitioning "Permalink to this headline")

The kernel now has tiles of global memory by applying the `CtaTiler` to the full tensors and it also has tiles of shared memory by allocating appropriately. We now want to create an efficient way to copy one tile of global memory to our tile of shared memory. A trivial way to do this would be to use a single thread and copy each element.

```cpp
if (thread0()) {
  Tensor gA0 = gA(_,_,0);  // (BLK_M,BLK_K), the 0th tile
  for (int i = 0; i < size(sA); ++i) {
    sA(i) = gA0(i);
  }
}
```

This would work, but we have lots of threads to use inside this CTA, so let’s use them!

If we partition the two tiles of data across the threads in the CTA, then each thread can copy its own subtensor of data. There are lots of ways this partitioning could occur, however.

The `gemm_nt` function defines two layouts of _threads_ as

```c++
  // Define thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{},Int<8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{},Int<8>{}));   // (n,k) -> thr_idx
```

and the `gemm_tn` functions defines two layouts of _threads_ as

```c++
  // Define thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{},Int<8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{},Int<8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
```

Both cases happen to use 32x8 threads, which will be used to partition a 128x8 tile of gmem and smem data into a 4x1 subtensor for each thread. The only difference here is that `gemm_nt` uses M-major and N-major threads to match the order of data in global memory and `gemm_tn` uses K-major threads to match the order of data in global memory.

Again, the conditions on the thread layouts are checked inside the kernel.

```cpp
  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K
```

These thread layouts are then used to partition the global memory tensors data and shared memory tensors

```cpp
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);    // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);    // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);    // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);    // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));  // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));  // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));  // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));  // THR_K
```

where `local_partition` is a lot like `local_tile`, except the coordinate slices into the tile-mode (the first mode) of the `zipped_divide` rather than the rest-mode (the second mode). That is, each thread gets one element of data assigned to it per thread tile and that thread tile is repeated to cover the entire data tile.

The naming convention `tAsA` is pretty typical across CuTe and CUTLASS. This is read as “Partitioning pattern `tA` applied to tensor `sA`”. In the next section, we’ll see a different partitioner applied to `sA` to produce `tCsA`. By applying the same partitioning pattern, `tA`, to tensors `sA` and `gA`, we preserve the _logical consistency_ of those tensors (checked by the assertions above) where logical elements between the two tensors correspond despite any differences in their data layouts. When used in `cute::copy`, for example, this naming convention let’s us lexically verify that the two tensors are using the same partitioning pattern.

With the data partitioned across the threads, _every thread_ can now participate in the copy by writing

```cpp
copy(tAgA(_,_,0), tAsA);
```

because every thread owns a different subtensor of the tile that will be copied.
