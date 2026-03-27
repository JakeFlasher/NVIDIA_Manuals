---
title: "5.5.9.2. Single-Precision-Only Intrinsic Functions"
section: "5.5.9.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#single-precision-only-intrinsic-functions"
---

### [5.5.9.2. Single-Precision-Only Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#single-precision-only-intrinsic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#single-precision-only-intrinsic-functions "Permalink to this headline")

The following table lists the single-precision floating-point intrinsic functions with their maximum ULP error.

- The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the _round-to-nearest ties-to-even_ rounding mode.
- The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.

| Function | Meaning | Maximum ULP Error |
| --- | --- | --- |
| [__fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff) | \(\(\dfrac{x}{y}\)\) | \(\(2\)\) for \(\(\|y\| \in [2^{-126}, 2^{126}]\)\) |
| [__frsqrt_rn(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv411__frsqrt_rnf) | \(\(\dfrac{1}{\sqrt{x}}\)\) | 0 ULP |
| [__expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff) | \(\(e^x\)\) | \(\(2 + \lfloor \|1.173 \cdot x\| \rfloor\)\) |
| [__exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff) | \(\(10^x\)\) | \(\(2 + \lfloor \|2.97 \cdot x\| \rfloor\)\) |
| [__powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff) | \(\(x^y\)\) | Derived from `exp2f(y * __log2f(x))` |
| [__logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff) | \(\(\ln(x)\)\) | ▪ \(\(2^{-21.41}\)\) abs error for \(\(x \in [0.5, 2]\)\) <br> ▪ 3 ULP, otherwise |
| [__log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff) | \(\(\log_2(x)\)\) | ▪ \(\(2^{-22}\)\) abs error for \(\(x \in [0.5, 2]\)\) <br> ▪ 2 ULP, otherwise |
| [__log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff) | \(\(\log_{10}(x)\)\) | ▪ \(\(2^{-24}\)\) abs error for \(\(x \in [0.5, 2]\)\) <br> ▪ 3 ULP, otherwise |
| [__sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff) | \(\(\sin(x)\)\) | ▪ \(\(2^{-21.41}\)\) abs error for \(\(x \in [-\pi, \pi]\)\) <br> ▪ larger otherwise |
| [__cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff) | \(\(\cos(x)\)\) | ▪ \(\(2^{-21.41}\)\) abs error for \(\(x \in [-\pi, \pi]\)\) <br> ▪ larger otherwise |
| [__sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf) | \(\(\sin(x), \cos(x)\)\) | Component-wise, the same as `__sinf(x)` and `__cosf(x)` |
| [__tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff) | \(\(\tan(x)\)\) | Derived from `__sinf(x) * (1 / __cosf(x))` |
| [__tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff) | \(\(\tanh(x)\)\) | ▪ Max relative error: \(\(2^{-11}\)\) <br> ▪ Subnormal results are not flushed to zero even under `-ftz=true` compiler flag. |
