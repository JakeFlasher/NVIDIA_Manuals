---
title: "GETT as GEMM"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#gett-as-gemm"
---

## [GETT as GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#gett-as-gemm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#gett-as-gemm "Permalink to this headline")

“GETT” here stands for “general(ized) tensor times tensor,” a tensor contraction.

CuTe permits matrices to have nested `Layout`s.
This means that we can fold a `Tensor` into a “matrix” by grouping modes according to their categories.

As a result, we can implement GETT by using
our existing GEMM implementation. Included below is a launcher like `gemm_nt` that uses the same device kernel contained in `sgemm_1.cu` to compute a GETT with two m-modes.

```cpp
// Setup params for a GETT with two m-modes.
// The A and C tensors are assumed to be m0-major.
//   Calls sgemm_1.cu's gemm_device<<<>>> without modification.
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gett(int m0, int m1, int n, int k,
     Alpha alpha,
     TA const* A, int ldAm1, int ldAk,  // m0-major
     TB const* B, int ldBk,
     Beta beta,
     TC      * C, int ldCm1, int ldCn,  // m0-major
     cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = make_shape(m0, m1);                               // (m0,m1)-multimode M
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(make_stride(Int<1>{}, ldAm1), ldAk); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(make_stride(Int<1>{}, ldCm1), ldCn); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Shape<_64, _2>{};    // Take _64 elements from m0 and _2 elements from m1
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}
```

Note that the only changes are the definition of shape `M`, the definition of strides `dA` and `dC`, and the definition of the CTA Tiler `bM`. The above uses a multimodel problem shape `M = (m0,m1)` and a multimodal CTA Tiler `bM = <_64,_2>` to change which portion of the global memory tensors `A` and `C` each CTA will be responsible for computing.

Similar examples can be found for CUTLASS 3.x kernels that are based on CuTe, such as [this Hopper GETT example](https://github.com/NVIDIA/cutlass/tree/main/examples/51_hopper_gett).
