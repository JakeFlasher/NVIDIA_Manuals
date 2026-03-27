---
title: "5.3.9.7. Templates"
section: "5.3.9.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#templates"
---

### [5.3.9.7. Templates](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#templates)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#templates "Permalink to this headline")

A type cannot be used as template argument of a `__global__` function or a `__device__/__constant__` variable (C++14) if either:

- The type is defined within a `__host__` or `__host__ __device__` function scope.
- The type is unnamed, such as an anonymous struct or a lambda expression, unless the type is local to a `__device__` or `__global__` function.
- The type is a class member with `private` or `protected`, unless the class is local to a `__device__` or `__global__` function.
- The type is compounded from any of the types above.

Example:

```cuda
template <typename T>
__global__ void kernel() {}

template <typename T>
__device__ int device_var; // C++14

struct {
    int v;
} unnamed_struct;

void host_function() {
    struct LocalStruct {};
//  kernel<LocalStruct><<<1, 1>>>(); // ERROR, LocalStruct is defined within a host function
    int data = 4;
//  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // ERROR, same as above

    auto lambda = [](){};
//  kernel<decltype(lambda)><<<1, 1>>>();         // ERROR, unnamed type
//  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // ERROR, unnamed type
}

class MyClass {
private:
    struct PrivateStruct {};
public:
    static void launch() {
//      kernel<PrivateStruct><<<1, 1>>>(); // ERROR, private type
    }
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/EhTn3GT3z).
