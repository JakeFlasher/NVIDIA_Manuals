---
title: "5.3.10.4. constexpr Variables"
section: "5.3.10.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#constexpr-variables"
---

### [5.3.10.4. constexpr Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#constexpr-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#constexpr-variables "Permalink to this headline")

By default, a `constexpr` variable cannot be used in a function with incompatible execution space, in the same way of standard variables.

A `constexpr` variable can be directly used in device code in the following cases:

- C++ scalar types, excluding pointer and pointer-to-member types:
  - `nullptr_t`.
  - `bool`.
  - Integral types: `char`, `signed char`, `unsigned`,  `long long`, etc.
  - Floating point types: `float`, `double`.
  - Enumerators:  `enum` and `enum class`.
- Class types:  `class`, `struct`, and `union` with a `constexpr` constructor.
- Raw array of the types above, for example  `int[]`, only when they are used inside a `constexpr` `__device__` or `__host__ __device__` function.

`constexpr __managed__` and `constexpr __shared__` variables are not allowed.

Examples:

```cuda
constexpr int ConstexprVar = 4; // scalar type

struct MyStruct {
    static constexpr int ConstexprVar = 100;
};

constexpr MyStruct my_struct = MyStruct{}; // class type

constexpr int array[] = {1, 2, 3};

__device__ constexpr int get_value(int idx) {
    return array[idx];                      // CORRECT
}

__device__ void foo(int idx) {
    int        v1 = ConstexprVar;           // CORRECT
    int        v2 = MyStruct::ConstexprVar; // CORRECT
//  const int &v3 = ConstexprVar1;          // ERROR, reference to host constexpr variable
//  const int *v4 = &ConstexprVar1;         // ERROR, address of host constexpr variable
    int        v5 = get_value(2);           // CORRECT, 'get_value(2)' is a constant expression.
//  int        v6 = get_value(idx);         // ERROR, 'get_value(idx)' is not a constant expression
//  int        v7 = array[2];               // ERROR, 'array' is not scalar type.
    MyStruct   v8 = my_struct;              // CORRECT
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/MWa1o3c9z).
