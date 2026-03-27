---
title: "5.3.9.4.1. Local Variables"
section: "5.3.9.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#local-variables"
---

#### [5.3.9.4.1. Local Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#local-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#local-variables "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__`, and `__constant__` memory space specifiers are not allowed on  non-`extern` variable declarations within a function that executes on the host.

Examples:

```cuda
__host__ void host_function() {
    int x;                   // CORRECT, __host__ variable
    __device__   int y;      // ERROR,   __device__ variable declaration within a host function
    __shared__   int z;      // ERROR,   __shared__ variable declaration within a host function
    __managed__  int w;      // ERROR,   __managed__ variable  declaration within a host function
    __constant__ int h;      // ERROR,   __constant__ variable declaration within a host function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

The `__device__`, `__constant__`, and `__managed__` memory space specifiers are not allowed on variable declarations that are neither `extern` nor `static` within a function that executes on the device.

```cuda
__device__ void device_function() {
    int x;                   // CORRECT, __device__ variable
    __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
    __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

see also the [static variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#static-variables) section.
