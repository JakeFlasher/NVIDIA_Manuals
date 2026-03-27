---
title: "2.1.9. Variable Specifiers"
section: "2.1.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#variable-specifiers"
---

## [2.1.9. Variable Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#variable-specifiers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#variable-specifiers "Permalink to this headline")

[CUDA specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-space-specifiers) can be used on static variable declarations to control placement.

- `__device__` specifies that a variable is stored in [Global Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-global-memory)
- `__constant__` specifies that a variable is stored in [Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-constant-memory)
- `__managed__` specifies that a variable is stored as [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory)
- `__shared__` specifies that a variable is store in [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory)

When a variable is declared with no specifier inside a `__device__` or `__global__` function, it is allocated to registers when possible, and [local memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-local-memory) when necessary. Any variable declared with no specifier outside a `__device__` or `__global__` function will be allocated in system memory.
