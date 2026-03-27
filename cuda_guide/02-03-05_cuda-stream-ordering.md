---
title: "2.3.5. CUDA Stream Ordering"
section: "2.3.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering"
---

## [2.3.5. CUDA Stream Ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#cuda-stream-ordering)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cuda-stream-ordering "Permalink to this headline")

Now that we have discussed the basic mechanisms of streams, events and callback functions it is important to consider the ordering semantics of asynchronous operations in a stream. These semantics are to allow application programmers to think about the ordering of operations in a stream in a safe way. There are some special cases where these semantics may be relaxed for purposes of performance optimization such as in the case of a _Programmatic Dependent Kernel Launch_ scenario, which allows the overlap of two kernels through the use of special attributes and kernel launch mechanisms, or in the case of batching memory transfers using the `cudaMemcpyBatchAsync()` function when the runtime can perform non-overlapping batch copies concurrently. We will discuss these optimizations later on _link needed_.

Most importantly CUDA streams are what are known as in-order streams. This means that the order of execution of the operations in a stream is the same as the order in which those operations were enqueued. An operation in a stream cannot leap-frog other operations. Memory operations (such as copies) are tracked by the runtime and will always complete before the next operation in order to allow dependent kernels safe access to the data being transferred.
