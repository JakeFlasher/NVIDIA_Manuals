---
title: "3.4.1.1. Device Enumeration"
section: "3.4.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#device-enumeration"
---

### [3.4.1.1. Device Enumeration](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#device-enumeration)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#device-enumeration "Permalink to this headline")

The following code sample shows how to query number of CUDA-enabled devices,
enumerate each of the devices, and query their properties.

```c++
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```
