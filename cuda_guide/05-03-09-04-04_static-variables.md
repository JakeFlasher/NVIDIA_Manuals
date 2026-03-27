---
title: "5.3.9.4.4. static Variables"
section: "5.3.9.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#static-variables"
---

#### [5.3.9.4.4. static Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#static-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#static-variables "Permalink to this headline")

`static` variables are allowed in device code in the following cases:

- Within `__global__` or `__device__`-only functions.
- Within `__host__ __device__` functions:
  - `static` variables without an explicit memory space (automatic deduction).
  - `static` variables with an explicit memory space, such as `static __device__/__constant__/__shared__/__managed__`, are allowed only when `__CUDA_ARCH__` is defined.

A `static` variable within a `__host__ __device__` function holds a different value depending on the execution space.

Examples of legal and illegal uses of function-scope `static` variables are shown below.

```cuda
struct TrivialStruct {
    int x;
};

struct NonTrivialStruct {
    __device__ NonTrivialStruct(int x) {}
};

__device__ void device_function(int x) {
    static int v1;              // CORRECT, implicit __device__ memory space specifier
    static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
//  static int v3 = x;           // ERROR, dynamic initialization is not allowed

    static __managed__  int v4; // CORRECT, explicit
    static __device__   int v5; // CORRECT, explicit
    static __constant__ int v6; // CORRECT, explicit
    static __shared__   int v7; // CORRECT, explicit

    static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
    static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
//  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
//  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/TdYKaTq3f).

---

```cuda
__host__ __device__ void host_device_function() {
    static            int v1; // CORRECT, implicit __device__ memory space specifier
//  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
#ifdef __CUDA_ARCH__
    static __device__ int v3; // CORRECT, declaration is only visible during device compilation
#else
    static int v4;            // CORRECT, declaration is only visible during host compilation
#endif
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/18qhjn8P1).

---

```cuda
#include <cassert>

__host__ __device__ int host_device_function() {
    static int v = 0;
    v++;
    return v;
}

__global__ void kernel() {
    int ret = host_device_function(); // v = 1
    assert(ret == 4);                 // FAIL
}

int main() {
    host_device_function();           // v = 1
    host_device_function();           // v = 2
    int ret = host_device_function(); // v = 3
    assert(ret == 3);                 // OK
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/Wqo9WjvYY).
