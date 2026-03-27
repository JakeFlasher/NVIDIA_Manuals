---
title: "2.1.4. Synchronizing CPU and GPU"
section: "2.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#synchronizing-cpu-and-gpu"
---

## [2.1.4. Synchronizing CPU and GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#synchronizing-cpu-and-gpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#synchronizing-cpu-and-gpu "Permalink to this headline")

As mentioned in [Launching Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-launching-kernels), kernel launches are asynchronous with respect to the CPU thread which called them. This means the control flow of the CPU thread will continue executing before the kernel has completed, and possibly even before it has launched. In order to guarantee that a kernel has completed execution before proceeding in host code, some synchronization mechanism is necessary.

The simplest way to synchronize the GPU and a host thread is with the use of `cudaDeviceSynchronize`, which blocks the host thread until all previously issued work on the GPU has completed. In the examples of this chapter this is sufficient because only single operations are being executed on the GPU. In larger applications, there may be multiple [streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams) executing work on the GPU and `cudaDeviceSynchronize` will wait for work in all streams to complete. In these applications, using [Stream Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#async-execution-stream-synchronization) APIs to synchronize only with a specific stream or [CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events) is recommended. These will be covered in detail in the [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution) chapter.
