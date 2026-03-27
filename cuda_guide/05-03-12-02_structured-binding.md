---
title: "5.3.12.2. Structured Binding"
section: "5.3.12.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#structured-binding"
---

### [5.3.12.2. Structured Binding](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#structured-binding)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#structured-binding "Permalink to this headline")

A structured binding cannot be declared with a memory space specifier, such as `__device__`, `__shared__`, `__constant__`, or `__managed__`.

Example:

```cuda
struct S {
    int x, y;
};
// __device__ auto [a, b] = S{4, 5}; // ERROR
```
