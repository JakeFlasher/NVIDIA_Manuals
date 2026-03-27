---
title: "5.4.11.2. Alternate Floating Point"
section: "5.4.11.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#alternate-floating-point"
---

### [5.4.11.2. Alternate Floating Point](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#alternate-floating-point)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#alternate-floating-point "Permalink to this headline")

Tensor Cores support alternate types of floating point operations on devices with compute capability 8.0 and higher.

**`__nv_bfloat16`**

  This data format is an alternate fp16 format that has the same range as f32 but reduced precision (7 bits). You can use this data format directly with the `__nv_bfloat16` type available in `cuda_bf16.h`. Matrix fragments with `__nv_bfloat16` data types are required to be composed with accumulators of `float` type. The shapes and operations supported are the same as with `__half`.

**`tf32`**

  This data format is a special floating-point format supported by Tensor Cores, with the same range as f32 and reduced precision (>=10 bits). The internal layout of this format is implementation-defined. To use this floating-point format with WMMA operations, the input matrices must be manually converted to tf32 precision.

  To facilitate conversion, a new intrinsic `__float_to_tf32` is provided. While the input and output arguments to the intrinsic are of `float` type, the output will be `tf32` numerically. This new precision is intended to be used with Tensor Cores only, and if mixed with other `float`type operations, the precision and range of the result will be undefined.

  Once an input matrix (`matrix_a` or `matrix_b`) is converted to tf32 precision, the combination of a `fragment` with `precision::tf32` precision, and a data type of `float` to `load_matrix_sync` will take advantage of this new capability. Both the accumulator fragments must have `float` data types. The only supported matrix size is 16x16x8 (m-n-k).

  The elements of the fragment are represented as `float`, hence the mapping from `element_type<T>` to `storage_element_type<T>` is:

  ```cuda
  precision::tf32 -> float
  ```
