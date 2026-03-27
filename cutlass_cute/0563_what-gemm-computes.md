---
title: "What gemm computes"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#what-gemm-computes"
---

### [What gemm computes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#what-gemm-computes)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#what-gemm-computes "Permalink to this headline")

The `gemm` algorithm takes three `Tensor`s, A, B, and C.
What it does depends on the number of modes
that its `Tensor` parameters have.
We express these modes using letters.

- V indicates a “vector,” a mode of independent elements.
- M and N indicate the number of rows resp. columns
of the matrix result C of the BLAS’s GEMM routine.
- K indicates the “reduction mode” of GEMM,
that is, the mode along which GEMM sums.
Please see the [GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) for details.

We list the modes of the input `Tensor`s A and B,
and the output `Tensor` C,
using a notation `(...) x (...) => (...)`.
The two leftmost `(...)` describe A and B (in that order),
and the `(...)` to the right of the `=>` describes C.

1. `(V) x (V) => (V)`. The element-wise product of vectors: C<sub>v</sub> += A<sub>v</sub> B<sub>v</sub>. Dispatches to FMA or MMA.
2. `(M) x (N) => (M,N)`. The outer product of vectors: C<sub>mn</sub> += A<sub>m</sub> B<sub>n</sub>. Dispatches to (4) with V=1.
3. `(M,K) x (N,K) => (M,N)`. The product of matrices: C<sub>mn</sub> += A<sub>mk</sub> B<sub>nk</sub>. Dispatches to (2) for each K.
4. `(V,M) x (V,N) => (V,M,N)`. The batched outer product of vectors: C<sub>vmn</sub> += A<sub>vm</sub> B<sub>vn</sub>. Optimizes for register reuse and dispatches to (1) for each M, N.
5. `(V,M,K) x (V,N,K) => (V,M,N)`. The batched product of matrices: C<sub>vmn</sub> += A<sub>vmk</sub> B<sub>vnk</sub>. Dispatches to (4) for each K.

Please refer to the [GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
for an overview of CuTe’s convention for ordering the modes.
For example, if K appears, it always appears rightmost (“outermost”).
If V appears, it always appears leftmost (“innermost”).
