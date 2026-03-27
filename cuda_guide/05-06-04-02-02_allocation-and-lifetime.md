---
title: "5.6.4.2.2. Allocation and Lifetime"
section: "5.6.4.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#allocation-and-lifetime"
---

#### [5.6.4.2.2. Allocation and Lifetime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#allocation-and-lifetime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#allocation-and-lifetime "Permalink to this headline")

`cudaMalloc()` and `cudaFree()` have distinct semantics between the host and device environments. When invoked from the host, `cudaMalloc()` allocates a new region from unused device memory. When invoked from the device runtime these functions map to device-side `malloc()` and `free()`. This implies that within the device environment the total allocatable memory is limited to the device `malloc()` heap size, which may be smaller than the available unused device memory. Also, it is an error to invoke `cudaFree()` from the host program on a pointer which was allocated by `cudaMalloc()` on the device or vice-versa.

|  | `cudaMalloc()` on Host | `cudaMalloc()` on Device |
| --- | --- | --- |
| `cudaFree()` on Host | Supported | Not Supported |
| `cudaFree()` on Device | Not Supported | Supported |
| Allocation limit | Available device memory | `cudaLimitMallocHeapSize` |
