---
title: "1.3.2.1. CUDA Runtime API and CUDA Driver API"
section: "1.3.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-runtime-api-and-cuda-driver-api"
---

### [1.3.2.1. CUDA Runtime API and CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#cuda-runtime-api-and-cuda-driver-api)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#cuda-runtime-api-and-cuda-driver-api "Permalink to this headline")

The CUDA runtime API is implemented on top of a lower-level API called the _CUDA driver API_, which is an API exposed by the NVIDIA Driver. This guide focuses on the APIs exposed by the CUDA runtime API. All the same functionality can be achieved using only the driver API if desired. Some features are only available using the driver API. Applications may use either API or both interoperably. Section [The CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api) covers interoperation between the runtime and driver APIs.

The full API reference for the CUDA runtime API functions can be found in the [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) .

The full API reference for the CUDA driver API can be found in the [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) .
