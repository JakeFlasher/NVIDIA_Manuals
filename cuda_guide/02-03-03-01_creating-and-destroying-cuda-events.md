---
title: "2.3.3.1. Creating and Destroying CUDA Events"
section: "2.3.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#creating-and-destroying-cuda-events"
---

### [2.3.3.1. Creating and Destroying CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#creating-and-destroying-cuda-events)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#creating-and-destroying-cuda-events "Permalink to this headline")

CUDA Events can be created and destroyed using the `cudaEventCreate()` and `cudaEventDestroy()` functions.

```c
cudaEvent_t event;

// Create the event
cudaEventCreate(&event);

// do some work involving the event

// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```

The application is responsible for destroying events when they are no longer needed.
