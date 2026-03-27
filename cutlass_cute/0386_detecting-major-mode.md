---
title: "Detecting major mode"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#detecting-major-mode"
---

### [Detecting major mode](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#detecting-major-mode)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#detecting-major-mode "Permalink to this headline")

Developers sometimes need to detect whether a tensor is MN-major or K-major.
(For definitions, see the [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html).)

- _Correct_: `cutlass::detail::is_major<0, Stride>()` or
`cutlass::detail::is_k_major()` from `include/cutlass/gemm/gemm.h`
- _Incorrect_: `get<0>(stride) == 1`

The second point is incorrect because it assumes that the mode
is a single integer, not a multimode.
This means that the code will fail to compile for tensor contractions.
For example, suppose that a tensor A
has shape `((X, Y), K)` and stride `((1, X), X*Y)`.
`get<0>(stride)` is the tuple `(1, X)`, not a single integer.
However, A is certainly M major if interpreted as a matrix.
