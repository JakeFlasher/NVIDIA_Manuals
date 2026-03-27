---
title: "5.6.4.4. Launch Setup APIs"
section: "5.6.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#launch-setup-apis"
---

### [5.6.4.4. Launch Setup APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#launch-setup-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#launch-setup-apis "Permalink to this headline")

[Device-Side Kernel Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#dynamic-parallelism-device-runtime-kernel-launch) describes the syntax for launching kernels from device code using the same triple chevron launch notation as the host CUDA Runtime API.

Kernel launch is a system-level mechanism exposed through the device runtime library. It is also available directly from PTX via `cudaGetParameterBuffer()` and `cudaLaunchDevice()` APIs. It is permitted for a CUDA application to call these APIs itself, with the same requirements as for PTX. In both cases, the user is then responsible for correctly populating all necessary data structures in the correct format according to specification. Backwards compatibility is guaranteed in these data structures.

As with host-side launch, the device-side operator `<<<>>>` maps to underlying kernel launch APIs. This allows users targeting PTX will to perform a launch. The NVCC compiler front-end translates `<<<>>>` into these calls.

| Runtime API Launch Functions | Description of Difference From Host Runtime Behavior (behavior is identical if no description) |
| --- | --- |
| `cudaGetParameterBuffer` | Generated automatically from `<<<>>>`. Note different API to host equivalent. |
| `cudaLaunchDevice` | Generated automatically from `<<<>>>`. Note different API to host equivalent. |

The APIs for these launch functions are different to those of the CUDA Runtime API, and are defined as follows:

```c++
extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```
