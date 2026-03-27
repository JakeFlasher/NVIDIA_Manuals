---
title: "2.3.3. CUDA Events"
section: "2.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events"
---

## [2.3.3. CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#cuda-events)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cuda-events "Permalink to this headline")

CUDA events are a mechanism for inserting markers into a CUDA stream. They are essentially like tracer particles that can be used to track the progress of tasks in a stream. Imagine launching two kernels into a stream. Without such tracking events, we would only be able to determine whether the stream is empty or not. If we had an operation that was dependent on the output of the first kernel, we would not be able to start that operation safely until we knew the stream was empty by which time both kernels would have completed.

Using CUDA Events we can do better. By enqueuing an event into a stream directly after the first kernel, but before the second kernel, we can wait for this event to come to the front of the stream. Then, we can safely start our dependent operation knowing that the first kernel has completed, but before the second kernel has started. Using CUDA events in this way can build up a graph of dependencies between operations and streams. This graph analogy translates directly into the later discussion of [CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#async-execution-cuda-graphs).

CUDA streams also keep time information which can be used to time kernel launches and memory transfers.
