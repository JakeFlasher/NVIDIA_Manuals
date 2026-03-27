---
title: "Asynchronous pipelines"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#asynchronous-pipelines"
---

### [Asynchronous pipelines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#asynchronous-pipelines)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#asynchronous-pipelines "Permalink to this headline")

In order to write a performant GEMM Kernel,
software pipelining is critical to hide the latency of global memory loads.
(Please refer to the
[Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining) document.)
Different threads or groups of threads
may have different roles in the pipeline.
Some are “producers” that load data or perform computations
to satisfy other threads’ input data dependencies.
The same or different threads may be “consumers”
that do other work with those input data dependencies,
once they are satisfied.
Starting with the Hopper architecture,
the presence of hardware-accelerated synchronization instructions
make it possible for “producer” and “consumer” threads
to communicate with each other efficiently
about their data dependencies.

Implementing a persistent GEMM algorithm calls for managing
dozens of different kinds of asynchronously executing operations
that synchronize using multiple barriers organized as a circular list.
This complexity is too much for human programmers to manage by hand.
As a result, we have developed
[asynchronous Pipeline classes](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/).
These classes help developers orchestrate a pipeline
of asynchronous producer and consumer threads,
without needing to worry about lower-level hardware details.
These classes serve a similar function as the various
[pipeline abstractions](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html)
in libcu++.
