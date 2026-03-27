---
title: "2.3.2.2. Launching Kernels in CUDA Streams"
section: "2.3.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-kernels-in-cuda-streams"
---

### [2.3.2.2. Launching Kernels in CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#launching-kernels-in-cuda-streams)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#launching-kernels-in-cuda-streams "Permalink to this headline")

The usual triple-chevron syntax for launching a kernel can also be used to launch a kernel into a specific stream. The stream is specified as an extra parameter to the kernel launch. In the following example the kernel named `kernel` is launched into the stream with handle `stream`, which is of type `cudaStream_t` and has been assumed to have been created previously:

```c
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

The kernel launch is asynchronous and the function call returns immediately. Assuming that the kernel launch is successful, the kernel will execute in the stream `stream` and the application is free to perform other tasks on the CPU or in other streams on the GPU while the kernel is executing.
