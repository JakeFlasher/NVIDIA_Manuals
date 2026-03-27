---
title: "2.3.10. Summary of Asynchronous Execution"
section: "2.3.10"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#summary-of-asynchronous-execution"
---

## [2.3.10. Summary of Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#summary-of-asynchronous-execution)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#summary-of-asynchronous-execution "Permalink to this headline")

The key points of this section are:

> - Asynchronous APIs allow us to express concurrent execution of tasks providing the way to express overlapping of various operations. The actual concurrency achieved is dependent on available hardware resources and compute-capabilities.
> - The key abstractions in CUDA for asynchronous execution are streams, events and callback functions.
> - Synchronization is possible at the event, stream and device level
> - The default stream is a blocking stream which synchronizes with all other blocking streams, but does not synchronize with non-blocking streams
> - The default stream behavior can be avoided using per-thread default streams via the `--default-stream per-thread` compiler option or the CUDA_API_PER_THREAD_DEFAULT_STREAM preprocessor macro.
> - Streams can be created with different priorities, which are hints to the runtime and may not be respected for memory transfers.
> - CUDA provides API functions to reduce, or overlap overheads of kernel launches and memory transfers such as CUDA Graphs, Batched Memory Transfers and Programmatic Dependent Kernel Launch.
