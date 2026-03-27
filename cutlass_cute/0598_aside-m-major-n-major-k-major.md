---
title: "Aside: M-major, N-major, K-major"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#aside-m-major-n-major-k-major"
---

#### [Aside: M-major, N-major, K-major](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#aside-m-major-n-major-k-major)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#aside-m-major-n-major-k-major "Permalink to this headline")

We’ve found that the BLAS convention of using “non-transposed” (N) and “transposed” (T) flags in conjunction with the mode conventions of `MxK * KxN` to confuse the core issue of “what layout does this matrix use” and “in which mode does my matrix have a stride-1?”. Indeed, the answer to those questions can always be found by inspecting the CuTe `Layout`.

Instead of row-major or column-major (or Transposed
and Not-Transposed), we have found it much more convenient to say that a matrix is “M-major” if it is stride-1 in the M-mode, “N-major” if it is stride-1 in the N-mode, or “K-major” if it is stride-1 in the K-mode. Furthermore, knowing that matrix multiply always performs a reduction in the K-mode, it is very convenient from a software perspective to always have the K-mode in the same place and adopt the mode convention `MxK * NxK`. Implementations will always reduce over the second mode (the K mode) of both input matrices and leads to cases where implementations can treat both input matrices the same way.

How do we translate this into the BLAS user’s experience?

| BLAS | A Majorness | A Layout | B Majorness | B Layout |
| --- | --- | --- | --- | --- |
| NT | M-major | `(M,K):(1,ldA)` | N-major | `(N,K):(1,ldB)` |
| TN | K-major | `(M,K):(ldA,1)` | K-major | `(N,K):(ldB,1)` |
| NN | M-major | `(M,K):(1,ldA)` | K-major | `(N,K):(ldB,1)` |
| TT | K-major | `(M,K):(ldA,1)` | N-major | `(N,K):(1,ldB)` |

Regardless, we’ll still use the BLAS “NT” and “TN” notations for high-level descriptions of kernels when it’s appropriate.
