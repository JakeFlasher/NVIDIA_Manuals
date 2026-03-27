---
title: "2.3.2.1. Creating and Destroying CUDA Streams"
section: "2.3.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#creating-and-destroying-cuda-streams"
---

### [2.3.2.1. Creating and Destroying CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#creating-and-destroying-cuda-streams)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#creating-and-destroying-cuda-streams "Permalink to this headline")

CUDA streams can be created using the `cudaStreamCreate()` function. The function call initializes the stream handle which can be used to identify the stream in subsequent function calls.

```c
cudaStream_t stream;        // Stream handle
cudaStreamCreate(&stream);  // Create a new stream

// stream based operations ...

cudaStreamDestroy(stream);  // Destroy the stream
```

If the device is still doing work in stream `stream` when the application calls `cudaStreamDestroy()`, the stream will complete all the work in the stream before being destroyed.
