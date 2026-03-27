---
title: "2.3.3.2. Inserting Events into CUDA Streams"
section: "2.3.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#inserting-events-into-cuda-streams"
---

### [2.3.3.2. Inserting Events into CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#inserting-events-into-cuda-streams)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#inserting-events-into-cuda-streams "Permalink to this headline")

CUDA Events can be inserted into a stream using the `cudaEventRecord()` function.

```c
cudaEvent_t event;
cudaStream_t stream;

// Create the event
cudaEventCreate(&event);

// Insert the event into the stream
cudaEventRecord(event, stream);
```
