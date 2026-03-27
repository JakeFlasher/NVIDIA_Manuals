---
title: "5.3.10.5. __global__ Variadic Template"
section: "5.3.10.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#global-variadic-template"
---

### [5.3.10.5. __global__ Variadic Template](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#global-variadic-template)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#global-variadic-template "Permalink to this headline")

A variadic `__global__` function template has the following restrictions:

- Only a single pack parameter is allowed.
- The pack parameter must be listed last in the template parameter list.

Examples:

```cuda
template <typename... Pack>
__global__ void kernel1(); // CORRECT

// template <typename... Pack, template T>
// __global__ void kernel2(); // ERROR, parameter pack is not the last parameter

template <typename... TArgs>
struct MyStruct {};

// template <typename... Pack1, typename... Pack2>
// __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // ERROR, more than one parameter pack
```

See the example on [Compiler Explorer](https://godbolt.org/z/x48KnPbbY).
