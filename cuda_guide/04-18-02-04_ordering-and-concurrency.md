---
title: "4.18.2.4. Ordering and Concurrency"
section: "4.18.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#ordering-and-concurrency"
---

### [4.18.2.4. Ordering and Concurrency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ordering-and-concurrency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ordering-and-concurrency "Permalink to this headline")

The ordering of kernel launches from the device runtime follows CUDA Stream ordering semantics. Within a grid, all kernel launches into the same stream (with the exception of [The Fire-and-Forget Stream](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#fire-and-forget-stream)) are executed in-order. With multiple threads in the same grid launching into the same stream, the ordering within the stream is dependent on the thread scheduling within the grid, which may be controlled with synchronization primitives such as `__syncthreads()`.

Note that while named streams are shared by all threads within a grid, the implicit _NULL_ stream is only shared by all threads within a thread block. If multiple threads in a thread block launch into the implicit stream, then these launches will be executed in-order. If threads in different thread blocks launch into the implicit stream, these launches may be executed concurrently. If concurrency is desired for launches from multiple threads within a thread block, explicit named streams should be used.

The device runtime introduces no new concurrency guarantees within the CUDA execution model. That is, there is no guarantee of concurrent execution between any number of different thread blocks on a device.

The lack of concurrency guarantee extends to a parent grid and their child grids. When a parent grid launches a child grid, the child may start to execute once stream dependencies are satisfied and hardware resources are available, but is not guaranteed to begin execution until the parent grid reaches an implicit synchronization point.

Concurrency may vary as a function of device configuration, application workload, and runtime scheduling. It is therefore unsafe to depend on any concurrency between different thread blocks.
