---
title: "2.1.7.1. Error State"
section: "2.1.7.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-state"
---

### [2.1.7.1. Error State](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#error-state)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#error-state "Permalink to this headline")

The CUDA runtime maintains a `cudaError_t` state for each host thread. The value defaults to `cudaSuccess` and is overwritten whenever an error occurs. `cudaGetLastError` returns current error state and then resets it to `cudaSuccess`. Alternatively, `cudaPeekLastError` returns error state without resetting it.

Kernel launches using [triple chevron notation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-launching-kernels-triple-chevron) do not return a `cudaError_t`. It is good practice to check the error state immediately after kernel launches to detect immediate errors in the kernel launch or [asynchronous errors](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-error-checking-asynchronous) prior to the kernel launch. A value of `cudaSuccess` when checking the error state immediately after a kernel launch does not mean the kernel has executed successfully or even started execution. It only verifies that the kernel launch parameters and execution configuration passed to the runtime did not trigger any errors and that the error state is not a previous or asynchronous error before the kernel started.
