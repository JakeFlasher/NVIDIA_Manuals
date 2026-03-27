---
title: "5.5.9.1. Basic Intrinsic Functions"
section: "5.5.9.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#basic-intrinsic-functions"
---

### [5.5.9.1. Basic Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#basic-intrinsic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#basic-intrinsic-functions "Permalink to this headline")

A subset of mathematical intrinsic functions allow specifying the rounding mode:

- Functions suffixed with `_rn` operate using the _round to nearest even_ rounding mode.
- Functions suffixed with `_rz` operate using the _round towards zero_ rounding mode.
- Functions suffixed with `_ru` operate using the _round up_ (toward positive infinity) rounding mode.
- Functions suffixed with `_rd` operate using the _round down_ (toward negative infinity) rounding mode.

The `__fadd_[rn,rz,ru,rd]()`, `__dadd_[rn,rz,ru,rd]()`,  `__fmul_[rn,rz,ru,rd]()`, and `__dmul_[rn,rz,ru,rd]()` functions map to addition and multiplication operations that the compiler never merges into the `FFMA` or `DFMA` instructions. In contrast, additions and multiplications generated from the `*` and `+` operators are often combined into `FFMA` or `DFMA`.

The following table lists the single- and double-precision floating-point intrinsic functions. All of them have a maximum ULP error of 0 and are IEEE-compliant.

| Meaning | `float` | `double` |
| --- | --- | --- |
| \(\(x + y\)\) | [__fadd_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fadd_rnff) | [__dadd_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dadd_rndd) |
| \(\(x - y\)\) | [__fsub_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fsub_rnff) | [__dsub_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dsub_rndd) |
| \(\(x \cdot y\)\) | [__fmul_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmul_rnff) | [__dmul_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dmul_rndd) |
| \(\(x \cdot y + z\)\) | [__fmaf_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff) | [__fma_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv48__fma_rnddd) |
| \(\(\dfrac{x}{y}\)\) | [__fdiv_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fdiv_rnff) | [__ddiv_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__ddiv_rndd) |
| \(\(\dfrac{1}{x}\)\) | [__frcp_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__frcp_rnf) | [__drcp_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__drcp_rnd) |
| \(\(\sqrt{x}\)\) | [__fsqrt_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fsqrt_rnf) | [__dsqrt_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv410__dsqrt_rnd) |
