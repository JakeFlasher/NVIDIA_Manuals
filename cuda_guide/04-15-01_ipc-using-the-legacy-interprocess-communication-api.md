---
title: "4.15.1. IPC using the Legacy Interprocess Communication API"
section: "4.15.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#ipc-using-the-legacy-interprocess-communication-api"
---

## [4.15.1. IPC using the Legacy Interprocess Communication API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ipc-using-the-legacy-interprocess-communication-api)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ipc-using-the-legacy-interprocess-communication-api "Permalink to this headline")

To share device memory pointers and events across processes, an application must use the CUDA Interprocess Communication API, which is described in detail in the reference manual.
The IPC API permits an application to get the IPC handle for a given device memory pointer using `cudaIpcGetMemHandle()`.
A CUDA IPC handle can be passed to another process using standard host operating system IPC mechanisms, e.g., interprocess shared memory or files.
`cudaIpcOpenMemHandle()` uses the IPC handle to retrieve a valid device pointer that can be used within the other process.
Event handles can be shared using similar entry points.

An example of using the IPC API is where a single primary process generates a batch of input data, making the data available to multiple secondary processes without requiring regeneration or copying.

> **Note**
>
> The IPC API is only supported on Linux.
>
> Note that the IPC API is not supported for `cudaMallocManaged` allocations.
>
> Applications using CUDA IPC to communicate with each other should be compiled, linked, and run with the same CUDA driver and runtime.
>
> Allocations made by `cudaMalloc()` may be sub-allocated from a larger block of memory for performance reasons. In such case, CUDA IPC APIs will share the entire underlying memory block which may cause other sub-allocations to be shared, which can potentially lead to information disclosure between processes. To prevent this behavior, it is recommended to only share allocations with a 2MiB aligned size.
>
> Only the IPC events-sharing APIs are supported on L4T and embedded Linux Tegra devices with compute capability 7.x and higher. The IPC memory-sharing APIs are not supported on Tegra platforms.
