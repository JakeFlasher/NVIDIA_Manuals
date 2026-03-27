---
title: "5.3.7.2. Extended Lambdas"
section: "5.3.7.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambdas"
---

### [5.3.7.2. Extended Lambdas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#extended-lambdas)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#extended-lambdas "Permalink to this headline")

| The `nvcc` flag `--extended-lambda` allows explicit annotations of execution spaces in a lambda expression. These annotations should appear after the lambda introducer and before the optional lambda declarator.
| `nvcc` defines the macro `__CUDACC_EXTENDED_LAMBDA__` when the `--extended-lambda` flag is specified.

- An _extended lambda_ is defined within the scope of an immediate or nested block of a `__host__` or `__host__ __device__` function.
- An _extended device lambda_ is a lambda expression annotated with the `__device__` keyword.
- An _extended host-device lambda_ is a lambda expression annotated with the `__host__ __device__` keywords.

Unlike standard lambda expressions, extended lambdas can be used as type arguments in `__global__` functions.

Example:

```cuda
void host_function() {
    auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
    auto lambda2 = [] __device__ {};           // extended lambda
    auto lambda3 = [] __host__ __device__ {};  // extended lambda
    auto lambda4 = [] __host__ {};             // NOT an extended lambda
}

__host__ __device__ void host_device_function() {
    auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
    auto lambda2 = [] __device__ {};           // extended lambda
    auto lambda3 = [] __host__ __device__ {};  // extended lambda
    auto lambda4 = [] __host__ {};             // NOT an extended lambda
}

__device__ void device_function() {
    // none of the lambdas within this function are extended lambdas,
    // because the enclosing function is not a __host__ or __host__ __device__  function.
    auto lambda1 = [] {};
    auto lambda2 = [] __device__ {};
    auto lambda3 = [] __host__ __device__ {};
    auto lambda4 = [] __host__ {};
}

auto global_lambda = [] __host__ __device__ { }; // NOT an extended lambda because it is not defined
                                                 // within a __host__ or __host__ __device__ function
```
