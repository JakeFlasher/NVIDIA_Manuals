---
title: "4.20.1. Introduction"
section: "4.20.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access--introduction"
---

## [4.20.1. Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#introduction)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#introduction "Permalink to this headline")

The `Driver Entry Point Access APIs` provide a way to retrieve the address of a CUDA driver function. Starting from CUDA 11.3, users can call into available CUDA driver APIs using function pointers obtained from these APIs.

These APIs provide functionality similar to their counterparts, dlsym on POSIX platforms and GetProcAddress on Windows. The provided APIs will let users:

- Retrieve the address of a driver function using the `CUDA Driver API.`
- Retrieve the address of a driver function using the `CUDA Runtime API.`
- Request _per-thread default stream_ version of a CUDA driver function. For more details, see [Retrieve Per-thread Default Stream Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#retrieve-per-thread-default-stream-versions).
- Access new CUDA features on older toolkits but with a newer driver.
