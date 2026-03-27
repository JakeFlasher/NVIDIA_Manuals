---
title: "5.3.10.6. Defaulted Functions = default"
section: "5.3.10.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#defaulted-functions-default"
---

### [5.3.10.6. Defaulted Functions = default](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#defaulted-functions-default)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#defaulted-functions-default "Permalink to this headline")

The CUDA compiler infers the execution space of explicitly-defaulted member functions as described in [Implicitly-declared and explicitly-defaulted functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compiler-generated-functions).

Execution space specifiers on explicitly-defaulted functions are ignored by the compiler, except in the case the function is defined out-of-line or is a `virtual` function.

Examples:

```cuda
struct MyStruct1 {
    MyStruct1() = default;
};

void host_function() {
    MyStruct1 my_struct; // __host__ __device__ constructor
}

__device__ void device_function() {
    MyStruct1 my_struct; // __host__ __device__ constructor
}

struct MyStruct2 {
    __device__ MyStruct2() = default; // WARNING: __device__ annotation is ignored
};

struct MyStruct3 {
    __host__ MyStruct3();
};
MyStruct3::MyStruct3() = default; // out-of-line definition, not ignored

__device__ void device_function2() {
//  MyStruct3 my_struct; // ERROR, __host__ constructor
}

struct MyStruct4 {
    //  MyStruct4::~MyStruct4 has host execution space, not ignored because virtual
    virtual __host__ ~MyStruct4() = default;
};

__device__ void device_function3() {
    MyStruct4 my_struct4;
    // implicit destructor call for 'my_struct4':
    //    ERROR: call from a __device__ function 'device_function3' to a
    //    __host__ function 'MyStruct4::~MyStruct4'
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/q1M4j8YYf).
