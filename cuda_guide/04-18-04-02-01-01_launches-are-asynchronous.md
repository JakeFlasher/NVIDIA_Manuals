---
title: "4.18.4.2.1.1. Launches are Asynchronous"
section: "4.18.4.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#launches-are-asynchronous"
---

##### [4.18.4.2.1.1. Launches are Asynchronous](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#launches-are-asynchronous)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#launches-are-asynchronous "Permalink to this headline")

Identical to host-side launches, all device-side kernel launches are asynchronous with respect to the launching thread. That is to say, the `<<<>>>` launch command will return immediately and the launching thread will continue to execute until it hits an implicit launch-synchronization point, such as at a kernel launched into the `cudaStreamTailLaunch` stream ([The Tail Launch Stream](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#tail-launch-stream)). The child grid may begin execution at any time after launch, but is not guaranteed to begin execution until the launching thread reaches an implicit launch-synchronization point.

Similar to host-side launch, work launched into separate streams may run concurrently, but actual concurrency is not guaranteed. Programs that depend upon concurrency between child kernels are not supported by the CUDA programming model and will have undefined behavior.
