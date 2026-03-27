---
title: "3.1.3.3. Implicit Synchronization"
section: "3.1.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#advanced-host-programming--implicit-synchronization"
---

### [3.1.3.3. Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#implicit-synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#implicit-synchronization "Permalink to this headline")

Two commands from different streams cannot run concurrently if any one of the following operations is issued in-between them by the host thread:

- a page-locked host memory allocation
- a device memory allocation
- a device memory set
- a memory copy between two addresses to the same device memory
- any CUDA command to the NULL stream
- a switch between the L1/shared memory configurations

Operations that require a dependency check include any other commands within the same stream as the launch being checked
and any call to `cudaStreamQuery()` on that stream. Therefore, applications should follow these guidelines to improve
their potential for concurrent kernel execution:

- All independent operations should be issued before dependent operations,
- Synchronization of any kind should be delayed as long as possible.
