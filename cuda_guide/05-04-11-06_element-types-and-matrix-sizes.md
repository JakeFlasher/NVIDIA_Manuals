---
title: "5.4.11.6. Element Types and Matrix Sizes"
section: "5.4.11.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#element-types-and-matrix-sizes"
---

### [5.4.11.6. Element Types and Matrix Sizes](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#element-types-and-matrix-sizes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#element-types-and-matrix-sizes "Permalink to this headline")

Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of `matrix_a`, `matrix_b` and `accumulator` matrix supported:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| __half | __half | float | 16x16x16 |
| __half | __half | float | 32x8x16 |
| __half | __half | float | 8x32x16 |
| __half | __half | __half | 16x16x16 |
| __half | __half | __half | 32x8x16 |
| __half | __half | __half | 8x32x16 |
| unsigned char | unsigned char | int | 16x16x16 |
| unsigned char | unsigned char | int | 32x8x16 |
| unsigned char | unsigned char | int | 8x32x16 |
| signed char | signed char | int | 16x16x16 |
| signed char | signed char | int | 32x8x16 |
| signed char | signed char | int | 8x32x16 |

Alternate floating-point support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| __nv_bfloat16 | __nv_bfloat16 | float | 16x16x16 |
| __nv_bfloat16 | __nv_bfloat16 | float | 32x8x16 |
| __nv_bfloat16 | __nv_bfloat16 | float | 8x32x16 |
| precision::tf32 | precision::tf32 | float | 16x16x8 |

Double Precision Support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| double | double | double | 8x8x4 |

Experimental support for sub-byte operations:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| precision::u4 | precision::u4 | int | 8x8x32 |
| precision::s4 | precision::s4 | int | 8x8x32 |
| precision::b1 | precision::b1 | int | 8x8x128 |
