---
title: "2.3.2.3. Launching Memory Transfers in CUDA Streams"
section: "2.3.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams"
---

### [2.3.2.3. Launching Memory Transfers in CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#launching-memory-transfers-in-cuda-streams)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#launching-memory-transfers-in-cuda-streams "Permalink to this headline")

To launch a memory transfer into a stream, we can use the function `cudaMemcpyAsync()`. This function is similar to the  `cudaMemcpy()` function, but it takes an additional parameter specifying the stream to use for the memory transfer. The function call in the code block below copies `size` bytes from the host memory pointed to by `src` to the device memory pointed to by `dst` in the stream `stream`.

```c
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

Like other asynchronous function calls, this function call returns immediately, whereas the `cudaMemcpy()` function blocks until the memory transfer is complete. In order to access the results of the transfer safely, the application must determine that the operation has completed using some form of synchronization.

Other CUDA memory transfer functions such as `cudaMemcpy2D()` also have asynchronous variants.

> **Note**
>
> In order for memory copies involving CPU memory to be carried out asynchronously, the host buffers must be pinned and page-locked. `cudaMemcpyAsync()` will function correctly if host memory which is not pinned and page-locked is used, but it will revert to a synchronous behavior which will not overlap with other work. This can inhibit the performance benefits of using asynchronous memory transfers. It is recommended programs use `cudaMallocHost()` to allocate buffers which will be used to send or receive data from GPUs.
