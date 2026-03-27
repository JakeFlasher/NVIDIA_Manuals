---
title: "5.3.9.6.2. Data Members"
section: "5.3.9.6.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#data-members"
---

#### [5.3.9.6.2. Data Members](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#data-members)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#data-members "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on `class`, `struct`, and `union` data members.

Only `static` data members evaluated at compile time are supported, such as [const-qualified](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#const-variables) and `constexpr` variables.

```cuda
struct MyStruct {
   static inline constexpr int value1 = 10; // C++17
   static constexpr        int value2 = 10; // C++11
   static const            int value3 = 10;
// static                  int value4; // ERROR
};
```
