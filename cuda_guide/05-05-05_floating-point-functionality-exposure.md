---
title: "5.5.5. Floating-Point Functionality Exposure"
section: "5.5.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-functionality-exposure"
---

## [5.5.5. Floating-Point Functionality Exposure](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#floating-point-functionality-exposure)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-functionality-exposure "Permalink to this headline")

The mathematical functions supported by CUDA are exposed through the following methods:

[Built-in C/C++ language arithmetic operators](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-math-operators):

- `x + y`,  `x - y`,  `x * y`,  `x / y`,  `x++`, `x--`, `x += y`, `x -= y`, `x *= y`, `x /= y`.
- Support single-, double-, and quad-precision types, `float`, `double`,  and `__float128/_Float128` respectively.
  - `__half` and `__nv_bfloat16` types are also supported by including the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.
  - `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#supported-floating-point-types) table.
- They are available in both host and device code.
- Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).

[CUDA C++ Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-cxx-standard-functions):

- Expose the full set of C++ `<cmath>` [header functions](https://en.cppreference.com/w/cpp/header/cmath) through the `<cuda/std/cmath>` header and the `cuda::std::` namespace.
- Support IEEE-754 standard floating-point types, `__half`, `float`, `double`, `__float128`, as well as  Bfloat16 `__nv_bfloat16`.
  - `__float128` support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#supported-floating-point-types) table.
- They are available in both host and device code.
- They often rely on the [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html). Therefore, there could be different levels of accuracy between the host and device code.
- Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#optimization-options).
- A subset of functionalities is also supported in constant expressions, such as `constexpr` functions, in accordance with the C++23 and C++26 standard specifications.

[CUDA C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-cxx-standard-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

- Expose a subset of the C `<math.h>` [header functions](https://en.cppreference.com/w/c/header/math.html).
- Support single and double-precision types, `float` and `double` respectively.
  - They are available in both host and device code.
  - They don’t require additional headers.
  - Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#optimization-options).
- A subset of the `<math.h>` header functionalities is also available for `__half`, `__nv_bfloat16`, and `__float128/_Float128` types. These functions have names that resemble those of the C Standard Library.
  - `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively. Their host and device code availability is defined on a per-function basis.
  - `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header and they are only available in device code.
- They can have a different accuracy between host and device code.

[Non-standard CUDA Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-additional-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

- Expose mathematical functionalities that are not part of the C/C++ Standard Library.
- Mainly support single- and double-precision types, `float` and `double` respectively.
  - Their host and device code availability is defined on a per-function basis.
  - They don’t require additional headers.
  - They can have a different accuracy between host and device code.
- `__nv_bfloat16`, `__half`, `__float128/_Float128` are supported for a limited set of functions.
  - `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.
  - `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header.
  - They are only available in device code.
- Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#optimization-options).

[Intrinsic Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-intrinsic-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

- Support single- and double-precision types, `float` and `double` respectively.
- They are only available in device code.
- They are faster but less accurate than the respective [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html).
- Their behavior is not affected by the `nvcc` [floating-point optimization flags](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#optimization-options) `-prec-div=false`, `-prec-sqrt=false`, and `-fmad=true`. The only exception is `-ftz=true`, which is also included in `-use_fast_math`.

| Functionality | Supported Types | Host | Device | Affected by Floating-Point Optimization Flags <br> (only for `float` and `double`) |
| --- | --- | --- | --- | --- |
| [Built-in C/C++ language arithmetic operators](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-math-operators) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128/_Float128`, `cuda::std::complex` | ✅ | ✅ | ✅ |
| [CUDA C++ Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128`, `cuda::std::complex` | ✅ | ✅ | ✅ |
| `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` ***** |  |  |  |  |
| [CUDA C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double` | ✅ | ✅ | ✅ |
| `__nv_bfloat16`, `__half` with limited support and similar names | On a per-function basis |  |  |  |
| `__float128/_Float128` with limited support and similar names | ❌ | ✅ |  |  |
| [Non-standard CUDA Mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-additional-functions) | `float`, `double` | On a per-function basis | ✅ |  |
| `__nv_bfloat16`, `__half`, `__float128/_Float128` with limited support | ❌ | ✅ |  |  |
| [Intrinsic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-intrinsic-functions) | `float`, `double` | ❌ | ✅ | Only with `-ftz=true`, also included in `-use_fast_math` |

***** The [CUDA C++ Standard Library functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cpp-standard-library) support queries for small floating-point types, such as [numeric_limits<T>](https://en.cppreference.com/w/cpp/types/numeric_limits.html), [fpclassify()](https://en.cppreference.com/w/cpp/numeric/math/fpclassify), [isfinite()](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html), [isnormal()](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html), [isinf()](https://en.cppreference.com/w/cpp/numeric/math/isinf.html), and [isnan()](https://en.cppreference.com/w/cpp/numeric/math/isnan.html).

The following sections provide accuracy information for some of these functions, when applicable. It uses ULP for quantification. For more information on the definition of the [Unit in the Last Place (ULP)](https://en.wikipedia.org/wiki/Unit_in_the_last_place), please see Jean-Michel Muller’s paper [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf).

---
