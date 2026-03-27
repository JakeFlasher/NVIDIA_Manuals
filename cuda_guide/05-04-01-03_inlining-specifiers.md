---
title: "5.4.1.3. Inlining Specifiers"
section: "5.4.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#inlining-specifiers"
---

### [5.4.1.3. Inlining Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#inlining-specifiers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inlining-specifiers "Permalink to this headline")

The following specifiers can be used to control inlining for `__host__` and `__device__` functions:

- `__noinline__`: Instructs  `nvcc` not to inline the function.
- `__forceinline__`: Forces `nvcc` to inline the function within a single translation unit.
- `__inline_hint__`: Enables aggressive inlining across translation units when using [Link-Time Optimization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-link-time-optimization).

These specifiers are mutually exclusive.
