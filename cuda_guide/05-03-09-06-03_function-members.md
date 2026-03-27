---
title: "5.3.9.6.3. Function Members"
section: "5.3.9.6.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#function-members"
---

#### [5.3.9.6.3. Function Members](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#function-members)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#function-members "Permalink to this headline")

`__global__` functions cannot be members of a `struct`, `class`, or `union`.

A `__global__` function is allowed in a `friend` declaration, but cannot be defined.

Example:

```cuda
struct MyStruct {
    friend __global__ void f();   // CORRECT, friend declaration only

//  friend __global__ void g() {} // ERROR, friend definition
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/rv6cP3b9j).
