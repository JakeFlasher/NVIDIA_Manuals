---
title: "2.3.7. Explicit Synchronization"
section: "2.3.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization"
---

## [2.3.7. Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#explicit-synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#explicit-synchronization "Permalink to this headline")

There are various ways to explicitly synchronize streams with each other.

`cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed.

`cudaStreamSynchronize()`takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device.

`cudaStreamWaitEvent()`takes a stream and an event as parameters (see [CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cuda-events) for a description of events)and makes all the commands added to the given stream after the call to `cudaStreamWaitEvent()`delay their execution until the given event has completed.

`cudaStreamQuery()`provides applications with a way to know if all preceding commands in a stream have completed.
