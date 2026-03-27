---
title: "5.3.9.5.5. __global__ Function Arguments Passing"
section: "5.3.9.5.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#global-function-arguments-passing"
---

#### [5.3.9.5.5. __global__ Function Arguments Passing](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#global-function-arguments-passing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#global-function-arguments-passing "Permalink to this headline")

When launching a `__global__` function [from device code](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-kernels), each argument must be trivially copyable and trivially destructible.

When a `__global__` function is launched from host code, each argument type may be non-trivially copyable or non-trivially destructible. However, the processing of these types does not follow the standard C++ model, as described below. The user code must ensure that this workflow does not affect program correctness. The workflow diverges from standard C++ in two areas:

1. **Raw memory copy instead of copy constructor invocation**

The CUDA Runtime passes the kernel arguments to the `__global__` function by copying the raw memory content, eventually using `memcpy`. If an argument is non-trivially copyable and provides a user-defined copy constructor, the operations and side effects of the invocation are skipped in the host-to-device copy.

Example:

```cuda
#include <cassert>
struct MyStruct {
    int  value = 1;
    int* ptr;
    MyStruct() = default;
    __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
};
__global__ void device_function(MyStruct my_struct) {
    // this assert fails because "my_struct" is obtained by copying
    // the raw memory content and the copy constructor is skipped.
    assert(my_struct.ptr == &my_struct.value); // FAIL
}
void host_function(MyStruct my_struct) {
    assert(my_struct.ptr == &my_struct.value); // CORRECT
}
int main() {
    MyStruct my_struct;
    host_function(my_struct);
    device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/xhqe16dec).
2. **Destructor may be invoked before the** `__global__` **function has finished**

Kernel launches are asynchronous with host execution. As a result, if a `__global__` function argument has a non-trivial destructor, the destructor may execute in host code even before the `__global__` function has finished execution. This may break programs where the destructor has side effects.

Example:

```cuda
#include <cassert>
__managed__ int var = 0;
struct MyStruct {
    __host__ __device__ ~MyStruct() { var = 3; }
};
__global__ void device_function(MyStruct my_struct) {
    assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
}
int main() {
    MyStruct my_struct;
    // GPU kernel execution is asynchronous with host execution.
    // As a result, MyStruct::~MyStruct() could be executed before
    // the kernel finishes executing.
    device_function<<<1, 1>>>(my_struct);
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/cn6Y5W6zs).
