---
title: "5.1.1. Obtain the GPU Compute Capability"
section: "5.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#obtain-the-gpu-compute-capability"
---

## [5.1.1. Obtain the GPU Compute Capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#obtain-the-gpu-compute-capability)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#obtain-the-gpu-compute-capability "Permalink to this headline")

The [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) page provides a comprehensive mapping from NVIDIA GPU models to their compute capability.

Alternatively, the [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) tool, provided with the [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/), can be used to get the compute capability of a GPU. For example, the following command will output the GPU names and compute capabilities available on the system:

```bash
nvidia-smi --query-gpu=name,compute_cap
```

At runtime, the compute capability can be obtained using the CUDA Runtime API [cudaDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151) , CUDA Driver API [cuDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266), or  NVML API [nvmlDeviceGetCudaComputeCapability()](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a):

```c++
#include <cuda_runtime_api.h>

int computeCapabilityMajor, computeCapabilityMinor;
cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);
```

```c++
#include <cuda.h>

int computeCapabilityMajor, computeCapabilityMinor;
cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);
```

```c++
#include <nvml.h> // required linking with -lnvidia-ml

int computeCapabilityMajor, computeCapabilityMinor;
nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);
```
