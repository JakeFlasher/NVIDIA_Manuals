---
title: "3.3.4. Interoperability between Runtime and Driver APIs"
section: "3.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#interoperability-between-runtime-and-driver-apis"
---

## [3.3.4. Interoperability between Runtime and Driver APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#interoperability-between-runtime-and-driver-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#interoperability-between-runtime-and-driver-apis "Permalink to this headline")

An application can mix runtime API code with driver API code.

If a context is created and made current via the driver API, subsequent runtime calls will use this context instead of creating a new one.

If the runtime is initialized, `cuCtxGetCurrent()` can be used to retrieve the context created during initialization. This context can be used by subsequent driver API calls.

The implicitly created context from the runtime is called the primary context (see [Runtime Initialization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-runtime-initialization)). It can be managed from the driver API with the [Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) functions.

Device memory can be allocated and freed using either API. `CUdeviceptr` can be cast to regular pointers and vice-versa:

```c++
CUdeviceptr devPtr;
float* d_data;

// Allocation using driver API
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;

// Allocation using runtime API
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;
```

In particular, this means that applications written using the driver API can invoke libraries written using the runtime API (such as cuFFT, cuBLAS, …).

All functions from the device and version management sections of the reference manual can be used interchangeably.
