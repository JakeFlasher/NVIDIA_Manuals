---
title: "5.5.2. Floating-Point Data Types"
section: "5.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-data-types"
---

## [5.5.2. Floating-Point Data Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#floating-point-data-types)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-data-types "Permalink to this headline")

CUDA supports the Bfloat16, half-, single-, double-, and quad-precision floating-point data types.
The following table summarizes the supported floating-point data types in CUDA and their requirements.

| Precision / Name | Data Type | IEEE-754 | Header / Built-in | Requirements |
| --- | --- | --- | --- | --- |
| Bfloat16 | `__nv_bfloat16` | ❌ | `<cuda_bf16.h>` | Compute Capability 8.0 or higher. |
| Half Precision | `__half` | ✅ | `<cuda_fp16.h>` |  |
| Single Precision | `float` | ✅ | Built-in |  |
| Double Precision | `double` | ✅ | Built-in |  |
| Quad Precision | `__float128`/`_Float128` | ✅ | Built-in   `<crt/device_fp128_functions.h>` for mathematical functions | Host compiler support and Compute Capability 10.0 or higher.   The C or C++ spelling, `_Float128` and `__float128` respectively, also depends on the host compiler support. |

CUDA also supports [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (`TF32`), [microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) floating-point types, and other [lower precision numerical formats](https://resources.nvidia.com/en-us-blackwell-architecture) that are not intended for general-purpose computation, but rather for specialized purposes involving tensor cores. These include 4-, 6-, and 8-bit floating-point types. See the [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html) for more details.

The following figure reports the mantissa and exponent sizes of the supported floating-point data types.

![Floating-Point Types: Mantissa and Exponent sizes](images/________-_____-____-______1.png)

The following table reports the ranges of the supported floating-point data types.

| Precision / Name | Largest Value | Smallest Positive Value | Smallest Positive Denormal | Epsilon |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Bfloat16 | \(\(\approx 2^{128}\)\) | \(\(\approx 3.39 \cdot 10^{38}\)\) | \(\(2^{-126}\)\) | \(\(\approx 1.18 \cdot 10^{-38}\)\) | \(\(2^{-133}\)\) | \(\(2^{-7}\)\) |
| Half Precision | \(\(\approx 2^{16}\)\) | \(\(65504\)\) | \(\(2^{-14}\)\) | \(\(\approx 6.1 \cdot 10^{-5}\)\) | \(\(2^{-24}\)\) | \(\(2^{-10}\)\) |
| Single Precision | \(\(\approx 2^{128}\)\) | \(\(\approx 3.40 \cdot 10^{38}\)\) | \(\(2^{-126}\)\) | \(\(\approx 1.18 \cdot 10^{-38}\)\) | \(\(2^{-149}\)\) | \(\(2^{-23}\)\) |
| Double Precision | \(\(\approx 2^{1024}\)\) | \(\(\approx 1.8 \cdot 10^{308}\)\) | \(\(2^{-1022}\)\) | \(\(\approx 2.22 \cdot 10^{-308}\)\) | \(\(2^{-1074}\)\) | \(\(2^{-52}\)\) |
| Quad Precision | \(\(\approx 2^{16384}\)\) | \(\(\approx 1.19 \cdot 10^{4932}\)\) | \(\(2^{-16382}\)\) | \(\(\approx 3.36 \cdot 10^{-4932}\)\) | \(\(2^{-16494}\)\) | \(\(2^{-112}\)\) |

> **Hint**
>
> The [CUDA C++ Standard Library](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cpp-standard-library) provides `cuda::std::numeric_limits` in the `<cuda/std/limits>` header to query the properties and the ranges of the supported floating-point types, including [microscaling formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). See the [C++ reference](https://en.cppreference.com/w/cpp/types/numeric_limits.html) for the list of queryable properties.

****Complex numbers support:****

- The [CUDA C++ Standard Library](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cpp-standard-library) supports complex numbers with the [cuda::std::complex](https://en.cppreference.com/w/cpp/numeric/complex) type in the `<cuda/std/complex>` header. See also the [libcu++ documentation](https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html) for more details.
- CUDA also provides basic support for complex numbers with the `cuComplex` and `cuDoubleComplex` types in the `cuComplex.h` header.

---
