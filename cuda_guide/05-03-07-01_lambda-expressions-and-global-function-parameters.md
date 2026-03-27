---
title: "5.3.7.1. Lambda Expressions and __global__ Function Parameters"
section: "5.3.7.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#lambda-expressions-and-global-function-parameters"
---

### [5.3.7.1. Lambda Expressions and __global__ Function Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#lambda-expressions-and-global-function-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#lambda-expressions-and-global-function-parameters "Permalink to this headline")

A lambda expression or a closure type can only be used as an argument to a `__global__` function if its execution space is `__device__` or `__host__ __device__`. Global or namespace scope lambda expressions cannot be used as arguments in a `__global__` function.

Examples:

```cuda
template <typename T>
 __global__ void kernel(T input) {}

 __device__ void device_function() {
     // device kernel call requires separate compilation (-rdc=true flag)
     kernel<<<1, 1>>>([](){});
     kernel<<<1, 1>>>([] __device__() {});          // extended lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // extended lambda
 }

 auto global_lambda = [] __host__ __device__() {};

 void host_function() {
     kernel<<<1, 1>>>([] __device__() {});          // CORRECT, extended lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // CORRECT, extended lambda
 //  kernel<<<1, 1>>>([](){});                      // ERROR, closure type with host execution space
 //  kernel<<<1, 1>>>(global_lambda);               // ERROR, extended lambda, but at global scope
 }
```

See the example on [Compiler Explorer](https://godbolt.org/z/ajrsn5z5Y).
