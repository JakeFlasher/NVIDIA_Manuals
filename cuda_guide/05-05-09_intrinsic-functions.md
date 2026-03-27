---
title: "5.5.9. Intrinsic Functions"
section: "5.5.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#intrinsic-functions"
---

## [5.5.9. Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#intrinsic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#intrinsic-functions "Permalink to this headline")

Intrinsic mathematical functions are faster and less accurate versions of their corresponding [CUDA C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).

- They have the same name prefixed with `__`, such as `__sinf(x)`.
- They are only available in device code.
- They are faster because they map to fewer native instructions.
- The flag `--use_fast_math` automatically translates the corresponding [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) into intrinsic functions. See the [–use_fast_math Effect](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#use-fast-math) section for the full list of affected functions.
