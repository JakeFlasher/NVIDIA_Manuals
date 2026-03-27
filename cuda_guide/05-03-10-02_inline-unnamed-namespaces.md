---
title: "5.3.10.2. inline Unnamed Namespaces"
section: "5.3.10.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#inline-unnamed-namespaces"
---

### [5.3.10.2. inline Unnamed Namespaces](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#inline-unnamed-namespaces)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-unnamed-namespaces "Permalink to this headline")

The following entities cannot be declared in namespace scope within an `inline` unnamed namespace:

- `__global__` function.
- `__device__`, `__constant__`, `__managed__`, `__shared__` variables.
- Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.
