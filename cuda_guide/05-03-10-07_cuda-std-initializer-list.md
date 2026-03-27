---
title: "5.3.10.7. [cuda::]std::initializer_list"
section: "5.3.10.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cuda-std-initializer-list"
---

### [5.3.10.7. [cuda::]std::initializer_list](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-std-initializer-list)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-std-initializer-list "Permalink to this headline")

| By default, the CUDA compiler implicitly considers the member functions of `[cuda::]std::initializer_list` to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code.
| The `nvcc` flag `--no-host-device-initializer-list` disables this behavior; member functions of `[cuda::]std::initializer_list` will then be considered as `__host__` functions and will not be directly invocable from device code.

A `__global__` function cannot have a parameter of type `[cuda::]std::initializer_list`.

Example:

```cuda
#include <initializer_list>

__device__ void foo(std::initializer_list<int> in) {}

__device__ void bar() {
    foo({4,5,6}); // (a) initializer list containing only constant expressions.
    int i = 4;
    foo({i,5,6}); // (b) initializer list with at least one  non-constant element.
                  // This form may have better performance than (a).
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/xeah7r44T).
