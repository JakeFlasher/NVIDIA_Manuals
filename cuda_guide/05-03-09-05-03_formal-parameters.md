---
title: "5.3.9.5.3. Formal Parameters"
section: "5.3.9.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#formal-parameters"
---

#### [5.3.9.5.3. Formal Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#formal-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#formal-parameters "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on formal parameters.

```cuda
void device_function1(__device__ int x) { } // ERROR, __device__ parameter
void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter
```
