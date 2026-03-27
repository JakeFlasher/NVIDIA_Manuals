---
title: "4.18.4.2.4. Device Management"
section: "4.18.4.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#device-management"
---

#### [4.18.4.2.4. Device Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-management "Permalink to this headline")

Only the device on which a kernel is running will be controllable from that kernel. This means that device APIs such as `cudaSetDevice()` are not supported by the device runtime. The active device as seen from the GPU (returned from `cudaGetDevice()`) will have the same device number as seen from the host system. The `cudaDeviceGetAttribute()` call may request information about another device as this API allows specification of a device ID as a parameter of the call. Note that the catch-all `cudaGetDeviceProperties()` API is not offered by the device runtime - properties must be queried individually.
