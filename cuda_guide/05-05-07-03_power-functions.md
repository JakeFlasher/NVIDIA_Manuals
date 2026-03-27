---
title: "5.5.7.3. Power Functions"
section: "5.5.7.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#power-functions"
---

### [5.5.7.3. Power Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#power-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#power-functions "Permalink to this headline")

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for power functions are available in both host and device code only for `float` and `double` types.

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| <br> [pow(x, y)](https://en.cppreference.com/w/cpp/numeric/math/pow.html) | <br> \(\(x^y\)\) | <br> N/A | <br> N/A | [powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff) <br> <br> 4 ULP | [pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43powdd) <br> <br> 2 ULP | [__nv_fp128_pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_powgg) <br> <br> 1 ULP |
| <br> [sqrt(x)](https://en.cppreference.com/w/cpp/numeric/math/sqrt.html) | <br> \(\(\sqrt{x}\)\) | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hsqrtK13__nv_bfloat16) <br> <br> 0 ULP | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hsqrtK6__half) <br> <br> 0 ULP | [sqrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sqrtff) <br> <br> ▪ 0 ULP <br> ▪ 1 ULP with `--use_fast_math` | [sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sqrtd) <br> <br> 0 ULP | [__nv_fp128_sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sqrtg) <br> <br> 0 ULP |
| <br> [cbrt(x)](https://en.cppreference.com/w/cpp/numeric/math/cbrt.html) <br> <br> | <br> \(\(\sqrt[3]{x}\)\) | <br> N/A | <br> N/A | [cbrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45cbrtff) <br> <br> 1 ULP | [cbrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44cbrtd) <br> <br> 1 ULP | <br> N/A |
| <br> [hypot(x, y)](https://en.cppreference.com/w/cpp/numeric/math/hypot.html) | <br> \(\(\sqrt{x^2 + y^2}\)\) | <br> N/A | <br> N/A | [hypotf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46hypotfff) <br> <br> 3 ULP | [hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45hypotdd) <br> <br> 2 ULP | [__nv_fp128_hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_hypotgg) <br> <br> 1 ULP |

***** <sub>Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.</sub>
