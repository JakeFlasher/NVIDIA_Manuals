---
title: "3.4.1.2. Device Selection"
section: "3.4.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#device-selection"
---

### [3.4.1.2. Device Selection](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#device-selection)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#device-selection "Permalink to this headline")

A host thread can set the device it is currently operating on at any time by calling `cudaSetDevice()`.
Device memory allocations and kernel launches are made on the current device; streams and events are created in association with the currently set device.
Until a call to `cudaSetDevice()` is made by the host thread, the current device defaults to device 0.

The following code sample illustrates how setting the current device affects subsequent memory allocation and kernel execution operations.

```c++
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```
