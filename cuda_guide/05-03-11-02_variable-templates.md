---
title: "5.3.11.2. Variable Templates"
section: "5.3.11.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#variable-templates"
---

### [5.3.11.2. Variable Templates](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#variable-templates)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#variable-templates "Permalink to this headline")

A `__device__` or `__constant__` variable template cannot be `const`-qualified when using the Microsoft compiler.

Examples:

```cuda
// ERROR on Windows (non-portable), const-qualified
template <typename T>
__device__ const T var = 0;

 // CORRECT, ptr1 is not const-qualified
template <typename T>
__device__ const T* ptr1 = nullptr;

// ERROR on Windows (non-portable), ptr2 is const-qualified
template <typename T>
__device__ const T* const ptr2 = nullptr;
```

See the example on [Compiler Explorer](https://godbolt.org/z/8hM5Yh7db).
