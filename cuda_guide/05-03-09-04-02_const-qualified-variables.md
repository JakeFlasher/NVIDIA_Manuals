---
title: "5.3.9.4.2. const-qualified Variables"
section: "5.3.9.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#const-qualified-variables"
---

#### [5.3.9.4.2. const-qualified Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#const-qualified-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#const-qualified-variables "Permalink to this headline")

A `const`-qualified variable without memory space annotations (`__device__` or `__constant__`) declared at global, namespace, or class scope is considered to be a host variable. Device code cannot contain a reference or take the address of the variable.

The variable may be directly used in device code, if

- it has been initialized with a constant expression before the point of use,
- the type is not `volatile`-qualified, and
- it has one of the following types:
  - built-in integral type, or
  - built-in floating point type, except when the host compiler is Microsoft Visual Studio.

Starting with C++14, it is recommended to use `constexpr` or `inline constexpr` (C++17) variables instead of `const`-qualified ones. `constexpr` variables are not subject to the same type restrictions and can be utilized directly in device code.

`__managed__` variables don’t support `const`-qualified types.

Examples:

```cuda
const            int   ConstVar          = 10;
const            float ConstFloatVar     = 5.0f;
inline constexpr float ConstexprFloatVar = 5.0f; // C++17

struct MyStruct {
    static const            int   ConstVar          = 20;
//  static const             float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
    static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
};

extern const int ExternVar;

__device__ void foo() {
    int array1[ConstVar];                     // CORRECT
    int array2[MyStruct::ConstVar];           // CORRECT

    const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
    constexpr float var2 = ConstexprFloatVar; // CORRECT
//  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
//  int&            var4 = ConstVar;           // ERROR, reference to host variable
//  int*            var5 = &ConstVar;          // ERROR, address of host variable
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/eWG8KxK94).
