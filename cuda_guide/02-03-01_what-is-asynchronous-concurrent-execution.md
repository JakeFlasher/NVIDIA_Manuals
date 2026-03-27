---
title: "2.3.1. What is Asynchronous Concurrent Execution?"
section: "2.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#what-is-asynchronous-concurrent-execution"
---

## [2.3.1. What is Asynchronous Concurrent Execution?](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#what-is-asynchronous-concurrent-execution)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#what-is-asynchronous-concurrent-execution "Permalink to this headline")

CUDA allows concurrent, or overlapping, execution of multiple tasks, specifically:

- computation on the host
- computation on the device
- memory transfers from the host to the device
- memory transfers from the device to the host
- memory transfers within the memory of a given device
- memory transfers among devices

The concurrency is expressed via an asynchronous interface, where a dispatching function call or kernel launch returns immediately. Asynchronous calls usually return before the dispatched operation has completed and may return before the asynchronous operation has started. The application is then free to perform other tasks at the same time as the originally dispatched operation. When the final results of the initially dispatched operation are needed, the application must perform some form of synchronization to ensure that the operation in question has completed.  A typical example of a concurrent execution pattern is the overlapping of host and device memory transfers with computation and thus reducing or eliminating their overhead.

![Asynchronous Concurrent Execution with CUDA streams](images/w___-__-____________-__________-__________1.png)

Figure 17 Asynchronous COncurrent Execution with CUDA streams[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#asynchronous-concurrent-execution-with-cuda-streams "Link to this image")

In general, asynchronous interfaces typically provide three main ways to synchronize with the dispatched operation

- a **blocking approach**, where the application calls a function that blocks, or waits until the operation has completed
- a **non-blocking approach**, or polling approach where the application calls a function that returns immediately and supplies information about the status of the operation
- a **callback approach**, where a pre-registered function is executed when the operation has completed.

While the programming interfaces are asynchronous, the actual ability to carry out various  operations concurrently will depend on the version of CUDA and the compute capability of the hardware being used – these details will be left to a later section of this guide (see [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities)).

In [Synchronizing CPU and GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-synchronizing-the-gpu), the CUDA runtime function `cudaDeviceSynchronize()` was introduced, which is a blocking call which waits for all previously issued work to complete. The reason the `cudaDeviceSynchronize()` call was needed is because the kernel launch is asynchronous and returns immediately. CUDA provides an API for both blocking and non-blocking approaches to synchronization and even supports the use of host-side callback functions.

The core API components for asynchronous execution in CUDA are **CUDA Streams** and **CUDA Events**.
In the rest of this section we will explain how these elements can be used to express asynchronous execution
in CUDA.

A related topic is that of **CUDA Graphs**, which allow a graph of asynchronous operations to be defined up front, which
can then be executed repeatedly with minimal overhead. We cover CUDA Graphs in a very introductory level in section
[2.4.9.2 Introduction to CUDA Graphs with Stream Capture](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#async-execution-cuda-graphs), and a
more comprehensive discussion is provided in section [4.1 CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs).
