---
title: "3.1.3.2. Explicit Synchronization"
section: "3.1.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#advanced-host-programming--explicit-synchronization"
---

### [3.1.3.2. Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#explicit-synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#explicit-synchronization "Permalink to this headline")

As previously outlined, there are a number of ways that streams can synchronize with other streams. The following provides common methods at different levels of granularity:
- `cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed.
- `cudaStreamSynchronize()`takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device.
- `cudaStreamWaitEvent()`takes a stream and an event as parameters (see [CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events) for a description of events) and makes all the commands added to the given stream after the call to `cudaStreamWaitEvent()`delay their execution until the given event has completed.
- `cudaStreamQuery()`provides applications with a way to know if all preceding commands in a stream have completed.
