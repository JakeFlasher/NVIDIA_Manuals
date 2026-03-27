---
title: "5.3.12.1. inline Variables"
section: "5.3.12.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#inline-variables"
---

### [5.3.12.1. inline Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#inline-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-variables "Permalink to this headline")

| In a single translation unit, using an `inline` variable provides no additional functionality beyond a regular variable and does not provide any practical advantage.
| `nvcc` allows `inline` variables with `__device__`, `__constant__`, or `__managed__` memory space only in [Separate Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation) mode or for variables with internal linkage.

> **Note**
>
> When using `gcc/g++` host compiler, an `inline` variable declared with `__managed__` memory space specifier may not be visible to the debugger.

Examples:

```cuda
inline        __device__ int device_var1;  // CORRECT, when compiled in Separate Compilation mode (-rdc=true or -dc)
                                           // ERROR, when compiled in Whole Program Compilation mode

static inline __device__ int device_var2;  // CORRECT, internal linkage

namespace {

inline __device__ int device_var3;         // CORRECT, internal linkage

inline __shared__ int shared_var;          // CORRECT, internal linkage

static inline __device__ int device_var4;  // CORRECT, internal linkage

inline __device__ int device_var5;         // CORRECT, internal linkage

} // namespace
```

See the example on [Compiler Explorer](https://godbolt.org/z/oraqeGTzY).
