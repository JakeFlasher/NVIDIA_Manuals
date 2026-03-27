---
title: "5.6.4.6. API Reference"
section: "5.6.4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#api-reference"
---

### [5.6.4.6. API Reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#api-reference)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#api-reference "Permalink to this headline")

The portions of the CUDA Runtime API supported in the device runtime are detailed here. Host and device runtime APIs have identical syntax; semantics are the same except where indicated. The following table provides an overview of the API relative to the version available from the host.

| Runtime API Functions | Details |
| --- | --- |
| `cudaDeviceGetCacheConfig` |  |
| `cudaDeviceGetLimit` |  |
| `cudaGetLastError` | Last error is per-thread state, not per-block state |
| `cudaPeekAtLastError` |  |
| `cudaGetErrorString` |  |
| `cudaGetDeviceCount` |  |
| `cudaDeviceGetAttribute` | Will return attributes for any device |
| `cudaGetDevice` | Always returns current device ID as would be seen from host |
| `cudaStreamCreateWithFlags` | Must pass `cudaStreamNonBlocking` flag |
| `cudaStreamDestroy` |  |
| `cudaStreamWaitEvent` |  |
| `cudaEventCreateWithFlags` | Must pass `cudaEventDisableTiming` flag |
| `cudaEventRecord` |  |
| `cudaEventDestroy` |  |
| `cudaFuncGetAttributes` |  |
| `cudaMemcpyAsync` | Notes about all `memcpy/memset` functions:   - Only async `memcpy/set` functions are supported - Only device-to-device `memcpy` is permitted - May not pass in local or shared memory pointers |
| `cudaMemcpy2DAsync` |  |
| `cudaMemcpy3DAsync` |  |
| `cudaMemsetAsync` |  |
| `cudaMemset2DAsync` |  |
| `cudaMemset3DAsync` |  |
| `cudaRuntimeGetVersion` |  |
| `cudaMalloc` | May not call `cudaFree` on the device on a pointer created on the host, and vice-versa |
| `cudaFree` |  |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |  |
| `cudaOccupancyMaxPotentialBlockSize` |  |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` |  |
