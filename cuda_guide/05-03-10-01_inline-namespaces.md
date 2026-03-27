---
title: "5.3.10.1. inline Namespaces"
section: "5.3.10.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#inline-namespaces"
---

### [5.3.10.1. inline Namespaces](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#inline-namespaces)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-namespaces "Permalink to this headline")

It is not allowed to define one of the following entities within an `inline` namespace when another entity of the same name and type signature is defined in an enclosing namespace:

- `__global__` function.
- `__device__`, `__constant__`, `__managed__`, `__shared__` variables.
- Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.

Example:

```cuda
__device__ int my_var; // global scope

inline namespace NS {

__device__ int my_var; // namespace scope

} // namespace NS
```
