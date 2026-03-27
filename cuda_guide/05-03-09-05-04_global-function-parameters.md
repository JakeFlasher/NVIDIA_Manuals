---
title: "5.3.9.5.4. __global__ Function Parameters"
section: "5.3.9.5.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#global-function-parameters"
---

#### [5.3.9.5.4. __global__ Function Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#global-function-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#global-function-parameters "Permalink to this headline")

A `__global__` function has the following restrictions:

- It cannot have a variable number of arguments, namely the C ellipsis syntax `...` and the `va_list` type. C++11 variadic template is allowed, subject to the restrictions described in the [__global__ Variadic Template](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cpp11-variadic-template) section.
- Function parameters are passed to the device via [constant memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#constant-memory) and their total size is limited to 32,764 bytes.
- Function parameters cannot be pass-by-reference or by pass-by-rvalue reference.
- Function parameters cannot be of type `std::initializer_list`.
- Polymorphic class parameters (`virtual`) are considered undefined behavior.
- Lambda expressions and closure types are allowed, subject to the restrictions described in the [Lambda Expressions and __global__ Function Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#lambda-expressions-global) section.
