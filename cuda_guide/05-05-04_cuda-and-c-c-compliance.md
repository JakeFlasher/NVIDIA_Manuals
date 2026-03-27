---
title: "5.5.4. CUDA and C/C++ Compliance"
section: "5.5.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#cuda-and-c-c-compliance"
---

## [5.5.4. CUDA and C/C++ Compliance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-and-c-c-compliance)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-and-c-c-compliance "Permalink to this headline")

****Floating-Point Exceptions:****

Unlike the host implementation, the mathematical operators and functions supported in device code do not set the global `errno` variable nor report [floating-point exceptions](https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions) to indicate errors. Thus, if error diagnostic mechanisms are required, users should implement additional input and output screening for the functions.

****Undefined Behavior with Floating-Point Operations:****

Common conditions of undefined behavior for mathematical operations include:

- Invalid arguments to mathematical operators and functions:
  - Using an uninitialized floating-point variable.
  - Using a floating-point variable outside its lifetime.
  - Signed integer overflow.
  - Dereferencing an invalid pointer.
- Floating-point specific undefined behavior:
  - Converting a floating-point value to an integer type for which the result is not representable is undefined behavior. This also includes NaN and infinity.

Users are responsible for ensuring the validity of a CUDA program. Invalid arguments may result in undefined behavior and be subject to compiler optimizations.

Contrary to integer division by zero, floating-point division by zero is not undefined behavior and not subject to compiler optimizations; rather, it is implementation-specific behavior. C++ implementations that conform to [IEC-60559](https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html) (IEEE-754), including CUDA, produce infinity. Note that invalid floating-point operations produce NaN and should not be misinterpreted as undefined behavior. Examples include zero divided by zero and infinity divided by infinity.

****Floating-Point Literals Portability:****

Both C and C++ allow for the representation of floating-point values in either decimal or hexadecimal notation. Hexadecimal floating-point literals, which are supported in [C99](https://en.cppreference.com/w/c/language/floating_constant.html) and [C++17](https://en.cppreference.com/w/cpp/language/floating_literal.html), denote a real value in scientific notation that can be precisely expressed in base-2. However, this does not guarantee that the literal will map to an actual value stored in a target variable (see the next paragraph). Conversely, a decimal floating-point literal may represent a numeric value that cannot be expressed in base-2.

According to the [C++ standard rules](https://eel.is/c++draft/lex.fcon#3), hexadecimal and decimal floating-point literals are rounded to the nearest representable value, larger or smaller, chosen in an implementation-defined manner. This rounding behavior may differ between the host and the device.

```cpp
float f1 = 0.5f;    // 0.5, '0.5f' is a decimal floating-point literal
float f2 = 0x1p-1f; // 0.5, '0x1p-1f' is a hexadecimal floating-point literal
float f3 = 0.1f;
// f1, f2 are represented as 0 01111110 00000000000000000000000
// f3     is represented as  0 01111011 10011001100110011001101
```

The run-time and compile-time evaluations of the same floating-point expression are subject to the following portability issues:

- The run-time evaluation of a floating-point expression may be affected by the selected rounding mode, floating-point contraction (FMA) and reassociation compiler settings, as well as floating-point exceptions. Note that CUDA does not support floating-point exceptions and the [rounding mode](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-rounding) is set to _round-to-nearest-ties-to-even_ by default. Other rounding modes can be selected using [intrinsic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#mathematical-functions-appendix-intrinsic-functions).
- The compiler may use a higher-precision internal representation for constant expressions.
- The compiler may perform optimizations, such as constant folding, constant propagation, and common subexpression elimination, which can lead to a different final value or comparison result.

****C Standard Math Library Notes:****

The host implementations of common mathematical functions are mapped to [C Standard Math Library functions](https://en.cppreference.com/w/c/header/math.html) in a platform-specific way. These functions are provided by the host compiler and the respective host `libm`, if available.

- Functions not available from the host compilers are implemented in the `crt/math_functions.h` header file. For example, `erfinv()` is implemented there.
- Less common functions, such as `rhypot()` and `cyl_bessel_i0()`, are only available in the device code.

As previously mentioned, the host and device implementations of mathematical functions are independent. For more details on the behavior of these functions, please refer to the host implementation’s documentation.

---
