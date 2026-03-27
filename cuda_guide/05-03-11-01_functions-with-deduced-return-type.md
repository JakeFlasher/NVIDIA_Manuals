---
title: "5.3.11.1. Functions with Deduced Return Type"
section: "5.3.11.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#functions-with-deduced-return-type"
---

### [5.3.11.1. Functions with Deduced Return Type](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#functions-with-deduced-return-type)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#functions-with-deduced-return-type "Permalink to this headline")

A `__global__` function cannot have a deduced return type `auto`.

Introspection of the return type of a `__device__` function with a deduced return type is not allowed in host code.

> **Note**
>
> The CUDA frontend compiler changes the function declaration to have a `void` return type, before invoking the host compiler. This may break introspection of the deduced return type of the `__device__` function in host code. Thus, the CUDA compiler will issue a compile-time error for referencing such a deduced return type outside of device function bodies.

Examples:

```cuda
 __device__ auto device_function(int x) { // deduced return type
     return x;                            // decltype(auto) has the same behavior
 }

 __global__ void kernel() {
     int x = sizeof(device_function(2));         // CORRECT, device code scope
 }

 // const int size = sizeof(device_function(2)); // ERROR, return type deduction on host

 void host_function() {
 //  using T = decltype(device_function(2));     // ERROR, return type deduction on host
 }

void host_fn1() {
  // ERROR, referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // ERROR, referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// ERROR, referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct MyStruct { };

// ERROR, referenced outside device function bodies
struct S1_derived_t : MyStruct<decltype(fn1)> { };
```
