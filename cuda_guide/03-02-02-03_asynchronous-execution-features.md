---
title: "3.2.2.3. Asynchronous Execution Features"
section: "3.2.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-execution-features"
---

### [3.2.2.3. Asynchronous Execution Features](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#asynchronous-execution-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#asynchronous-execution-features "Permalink to this headline")

Recent NVIDIA GPU generations have included asynchronous execution capabilities to allow more overlap of data movement, computation, and synchronization within the GPU. These capabilities enable certain operations invoked from GPU code to execute asynchronously to other GPU code in the same thread block. This asynchronous execution should not be confused with asynchronous CUDA APIs discussed in [Section 2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution), which enable GPU kernel launches or memory operations to operate asynchronously to each other or to the CPU.

Compute capability 8.0 (The NVIDIA Ampere GPU Architecture) introduced hardware-accelerated asynchronous data copies from global to shared memory and asynchronous barriers (see [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) ).

Compute capability 9.0 (The NVIDIA Hopper GPU architecture) extended the asynchronous execution features with the [Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-async-copies) unit, which can transfer large blocks of data and multidimensional tensors from global memory to shared memory and vice versa, asynchronous transaction barriers, and asynchronous matrix multiply-accumulate operations (see [Hopper Architecture in Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) blog post for details.).

CUDA provides APIs which can be called by threads from device code to use these features. The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads.

An asynchronous operation is an operation initiated by a CUDA thread, but executed asynchronously as if by another thread, which we will refer to as an _async thread_. In a well-formed program, one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads. The async thread is always associated with the CUDA thread that initiated the operation.

An asynchronous operation uses a synchronization object to signal its completion, which could be a barrier or a pipeline. These synchronization objects are explained in detail in [Advanced Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-advanced-sync-primitives), and their role in performing asynchronous memory operations is demonstrated in [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#advanced-kernels-async-copies).
