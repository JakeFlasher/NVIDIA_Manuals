---
title: "5.5.7. CUDA C++ Mathematical Standard Library Functions"
section: "5.5.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#cuda-c-mathematical-standard-library-functions"
---

## [5.5.7. CUDA C++ Mathematical Standard Library Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-c-mathematical-standard-library-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-c-mathematical-standard-library-functions "Permalink to this headline")

CUDA provides comprehensive support for [C++ Standard Library mathematical functions](https://en.cppreference.com/w/cpp/header/cmath.html) through the `cuda::std::` namespace. The functionalities are part of the `<cuda/std/cmath>` header.
They are available in both host and device code.

The following sections specify the mapping with the [CUDA Math APIs](https://docs.nvidia.com/cuda/cuda-math-api/index.html) and the error bounds of each function when executed on the device.

- The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the _round-to-nearest ties-to-even_ rounding mode.
- The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.
