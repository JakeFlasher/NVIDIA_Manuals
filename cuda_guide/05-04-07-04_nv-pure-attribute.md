---
title: "5.4.7.4. __nv_pure__ Attribute"
section: "5.4.7.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-pure-attribute"
---

### [5.4.7.4. __nv_pure__ Attribute](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-pure-attribute)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-pure-attribute "Permalink to this headline")

In C/C++, a pure function has no side effects on its parameters and can access global variables, though it does not modify them.

CUDA provides `__nv_pure__` attribute supported for both host and device functions. The compiler translates `__nv_pure__` to the `pure` GNU attribute or to the Microsoft Visual Studio `noalias` attribute.

```cuda
__device__ __nv_pure__
int add(int a, int b) {
    return a + b;
}
```
