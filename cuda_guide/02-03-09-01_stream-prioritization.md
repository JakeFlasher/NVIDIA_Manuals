---
title: "2.3.9.1. Stream Prioritization"
section: "2.3.9.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#stream-prioritization"
---

### [2.3.9.1. Stream Prioritization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#stream-prioritization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#stream-prioritization "Permalink to this headline")

As mentioned previously, developers can assign priorities to CUDA streams. Prioritized streams need to be created using the `cudaStreamCreateWithPriority()` function. The function takes two parameters: the stream handle and the priority level.
The general scheme is that lower numbers correspond to higher priorities. The given priority range for a given device and context
can be queried using the `cudaDeviceGetStreamPriorityRange()` function. The default priority of a stream is 0.

```c
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```

We should note that a priority of a stream is only a hint to the runtime and generally applies primarily to kernel launches, and may not be respected for memory transfers. Stream priorities will not preempt already executing work, or guarantee any specific execution order.
