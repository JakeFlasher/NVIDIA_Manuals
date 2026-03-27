---
title: "4.3.4.1. Query for Support"
section: "4.3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#query-for-support"
---

### [4.3.4.1. Query for Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#query-for-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#query-for-support "Permalink to this headline")

An application can determine whether or not a device supports the stream-ordered
memory allocator by calling `cudaDeviceGetAttribute()` (see [developer blog](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/))
with the device attribute `cudaDevAttrMemoryPoolsSupported`.

IPC memory pool support can be queried with the
`cudaDevAttrMemoryPoolSupportedHandleTypes` device attribute. This attribute was added
in CUDA 11.3, and older drivers will return `cudaErrorInvalidValue` when this attribute
is queried.

```c++
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                           cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // `device` supports the Stream-Ordered Memory Allocator
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
              cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
   // Pools on the specified device can be created with posix file descriptor-based IPC
}
```

Performing the driver version check before the query avoids hitting a
`cudaErrorInvalidValue` error on drivers where the attribute was not yet
defined. One can use `cudaGetLastError` to clear the error instead of
avoiding it.
