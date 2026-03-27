---
title: "5.3.13.2. consteval Functions"
section: "5.3.13.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#consteval-functions"
---

### [5.3.13.2. consteval Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#consteval-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#consteval-functions "Permalink to this headline")

`consteval` functions can be called from both host and device code, independently of their execution space.

Examples:

```cuda
consteval int host_consteval() {
    return 10;
}

__device__ consteval int device_consteval() {
    return 10;
}

__device__ int device_function() {
    return host_consteval();   // CORRECT, even if called from device code
}

__host__ __device__ int host_device_function() {
    return device_function();  // CORRECT, even if called from host-device code
}
```
