---
title: "5.5.7.6. Error and Gamma Functions"
section: "5.5.7.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#error-and-gamma-functions"
---

### [5.5.7.6. Error and Gamma Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#error-and-gamma-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#error-and-gamma-functions "Permalink to this headline")

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for error and gamma functions are available in both host and device code for `float` and `double` types.

Error and Gamma functions are not natively available for CUDA-extended floating-point types, such as `__half` and `__nv_bfloat16`. In these cases, the functions are emulated by converting to a `float` type and then converting the result back.

| `cuda::std` Function | Meaning | `float` | `double` |
| --- | --- | --- | --- |
| <br> [erf(x)](https://en.cppreference.com/w/cpp/numeric/math/erf.html) | <br> \(\(\dfrac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt\)\) | [erff(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44erfff) <br> <br> 2 ULP | [erf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43erfd) <br> <br> 2 ULP |
| <br> [erfc(x)](https://en.cppreference.com/w/cpp/numeric/math/erfc.html) | <br> \(\(1 - \mathrm{erf}(x)\)\) | [erfcf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45erfcff) <br> <br> 4 ULP | [erfc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44erfcd) <br> <br> 5 ULP |
| <br> [tgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/tgamma.html) | <br> \(\(\Gamma(x)\)\) | [tgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47tgammaff) <br> <br> 5 ULP | [tgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46tgammad) <br> <br> 10 ULP |
| <br> [lgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/lgamma.html) | <br> \(\(\ln \|\Gamma(x)\|\)\) | [lgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lgammaff) <br> <br> ▪ 6 ULP for \(\(x \notin [-10.001, -2.264]\)\) <br> ▪ larger otherwise | [lgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lgammad) <br> <br> ▪ 4 ULP for \(\(x \notin [-23.0001, -2.2637]\)\) <br> ▪ larger otherwise |
