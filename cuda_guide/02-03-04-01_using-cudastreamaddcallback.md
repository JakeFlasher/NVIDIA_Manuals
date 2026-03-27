---
title: "2.3.4.1. Using cudaStreamAddCallback()"
section: "2.3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#using-cudastreamaddcallback"
---

### [2.3.4.1. Using cudaStreamAddCallback()](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#using-cudastreamaddcallback)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#using-cudastreamaddcallback "Permalink to this headline")

> **Note**
>
> The `cudaStreamAddCallback()` function is slated for deprecation and removal and is discussed here for completeness and because it may still appear in existing code. Applications should use or switch to using `cudaLaunchHostFunc()`.

The signature of the `cudaStreamAddCallback()` function is as follows:

```c
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

where

- `stream`: The stream to launch the callback function into.
- `callback`: The callback function to launch.
- `userData`: A pointer to the data to pass to the callback function.
- `flags`: Currently, this parameter must be 0 for future compatibility.

The signature of the  `callback` function is a little different from the case when we used the `cudaLaunchHostFunc()` function.
In this case the callback function is a C function with the signature:

```c
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```

where the function is now passed

- `stream`: The stream handle from which the callback function was launched.
- `status`: The status of the stream operation that triggered the callback.
- `userData`: A pointer to the data that was passed to the callback function.

In particular the `status` parameter will contain the current error status of the stream, which may have been set by previous operations.
Similarly to the `cudaLaunchHostFunc()` func case, the stream will not be active and advance to tasks until the host-function has completed, and no CUDA functions may be called from within the callback function.
