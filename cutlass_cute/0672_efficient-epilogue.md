---
title: "Efficient Epilogue"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#efficient-epilogue"
---

### [Efficient Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#efficient-epilogue)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#efficient-epilogue "Permalink to this headline")

CUTLASS GEMM operators perform mma followed by epilogue operation similar
to cuBLAS. CUTLASS implements an efficient row-major epilogue. Thus, to achieve
column-major GEMM, operands A & B are transposed and swapped.

To enable efficient row-major epilogue for both row-major and column-major output layout,
CUTLASS’ device-level GEMM operators `cutlass::device::Gemm` and `cutlass::device::GemmUniversal`
provide two template definitions:

- (a) [General definition](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm.h#L217)
- (b) [Specialized definition for column-major source/output](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm.h#L545)

Efficient row-major epilogue for:

- (i)  GEMM operator on row-major source/output uses template (a). It runs row-major GEMM and
an efficient row-major epilogue.
- (ii)  GEMM operator on column-major source/output uses template (b). It transposes and swaps
operands A and B to enable efficient epilogue. `A x B = C => Transpose(B) x Transpose(A) = Transpose(C)`.
For column-major source (C) matrix, Transpose(C) is row-major, and efficient epilogue works on
row-major.

Note that cuBLAS typically expects a column-major source (C) and output matrix (D). Thus,
CUTLASS library only instantiates and generates GEMM operatos with column-major layout. However,
CUTLASS by itself can run both row-major and column-major output layouts for all combinations
of input layouts. Thus, CUTLASS supports the following layout combinations for input and output layouts:

- `{N,T} x {N,T} => {N,T}` - NN, TN, TN, TT GEMM for both row-major and column-major output
