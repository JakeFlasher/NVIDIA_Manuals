---
title: "Mainloop"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#mainloop"
---

### [Mainloop](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#mainloop)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#mainloop "Permalink to this headline")

The mainloop iterates over tiles of global memory, reads those tiles into shared memory, and then performs the matrix-multiply and accumulates into the accumulators.

```c++
// TUTORIAL: Example of a very simple compute mainloop
//   copy(.) operates on the global and shared memory via the tA|tB partitioning
//   gemm(.) operates on the shared and register memory via the tC partitioning

auto K_TILE_MAX = size<2>(tAgA);

for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
{
  // Copy gmem to smem with tA|tB thread-partitioned tensors
  copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
  copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

  cp_async_fence();        // Label the end of (potential) cp.async instructions
  cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
  __syncthreads();         // Wait for all threads to write to smem

  // Compute gemm on tC thread-partitioned smem
  gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
  __syncthreads();         // Wait for all threads to read from smem
}
```

We can see that `k_tile` iterates over each tile of data, the `cute::copy` is performed for the current `k_tile` using the `tA` and `tB` thread-partitioned tensors, and the `cute::gemm` is computed for that current `k_tile` using the `tC` thread-partitioned tensors. Synchronization is provided so that this kernel works on any architecture.
