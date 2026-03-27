---
title: "5.5.6. Built-In Arithmetic Operators"
section: "5.5.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#built-in-arithmetic-operators"
---

## [5.5.6. Built-In Arithmetic Operators](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#built-in-arithmetic-operators)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-arithmetic-operators "Permalink to this headline")

The built-in C/C++ language operators, such as `x + y`, `x - y`, `x * y`, `x / y`, `x++`, `x--`, and reciprocal `1 / x`, for single-, double-, and quad-precision types comply with the IEEE-754 standard. They guarantee a maximum ULP error of zero using a _round-to-nearest-ties-to-even_ rounding mode. They are available in both host and device code.

The `nvcc` compilation flag `-fmad=true`, also included in `--use_fast_math`, enables contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations and has the following effect on the maximum ULP error for the single-precision type `float`:

- `x * y + z` → [__fmaf_rn(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff): 0 ULP

The `nvcc` compilation flag `-prec-div=false`, also included in `--use_fast_math`, has the following effect on the maximum ULP error for the division operator `/` for the single-precision type `float`:

- `x / y` → [__fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107): 2  ULP
- `1 / x`: 1 ULP

---
