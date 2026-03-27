---
title: "2.3.8. Implicit Synchronization"
section: "2.3.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#implicit-synchronization"
---

## [2.3.8. Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#implicit-synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#implicit-synchronization "Permalink to this headline")

Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted
in-between them, unless the streams are non-blocking streams (created with the `cudaStreamNonBlocking` flag).

Applications should follow these guidelines to improve their potential for concurrent kernel execution:

- All independent operations should be issued before dependent operations,
- Synchronization of any kind should be delayed as long as possible.
