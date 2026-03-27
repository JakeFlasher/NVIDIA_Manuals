---
title: "5.3.7. Lambda Expressions"
section: "5.3.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#lambda-expressions"
---

## [5.3.7. Lambda Expressions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#lambda-expressions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#lambda-expressions "Permalink to this headline")

| The compiler determines the execution space of a lambda expression or closure type (C++11) by associating it with the execution space of the innermost enclosing function scope. If there is no enclosing function scope, the execution space is specified as `__host__`.
| The execution space can also be specified explicitly with the [extended lambda syntax](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#extended-lambdas).

Examples:

```cuda
auto global_lambda = [](){ return 0; }; // __host__

void host_function() {
    auto lambda1 = [](){ return 1; };   // __host__
    [](){ return 3; };                  // __host__, closure type (body of a lambda expression)
}

__device__ void device_function() {
    auto lambda2 = [](){ return 2; };   // __device__
}

__global__ void kernel_function(void) {
    auto lambda3 = [](){ return 3; };   // __device__
}

__host__ __device__ void host_device_function() {
    auto lambda4 = [](){ return 4; };   // __host__ __device__
}

using function_ptr_t = int (*)();

__device__ void device_function(float          value,
                                function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}
```

See the example on [Compiler Explorer](https://godbolt.org/z/scv4vcczr).
