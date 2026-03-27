---
title: "2.3.6. Blocking and non-blocking streams and the default stream"
section: "2.3.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#blocking-and-non-blocking-streams-and-the-default-stream"
---

## [2.3.6. Blocking and non-blocking streams and the default stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#blocking-and-non-blocking-streams-and-the-default-stream)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#blocking-and-non-blocking-streams-and-the-default-stream "Permalink to this headline")

In CUDA there are two types of streams: blocking and non-blocking. The name can be a little misleading as the blocking and non-blocking semantics refer only to how the streams synchronize with the default stream. By default, streams created with `cudaStreamCreate()` are blocking streams. In order to create a non-blocking stream, the `cudaStreamCreateWithFlags()` function must be used with the `cudaStreamNonBlocking` flag:

```c
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

and non-blocking streams can be destroyed in the usual way with `cudaStreamDestroy()`.
