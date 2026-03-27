---
title: "3.2.2.3.1. Async Thread and Async Proxy"
section: "3.2.2.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#async-thread-and-async-proxy"
---

#### [3.2.2.3.1. Async Thread and Async Proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#async-thread-and-async-proxy)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#async-thread-and-async-proxy "Permalink to this headline")

Asynchronous operations may access memory differently than regular operations. To distinguish between these different memory access methods, CUDA introduces the concepts of an _async thread_, a _generic proxy_, and an _async proxy_. Normal operations (loads and stores) go through the generic proxy. Some asynchronous instructions, such as [LDGSTS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-ldgsts) and [STAS/REDAS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-stas), are modeled using an async thread operating in the generic proxy. Other asynchronous instructions, such as bulk-asynchronous copies with TMA and some tensor core operations (tcgen05.*, wgmma.mma_async.*), are modeled using an async thread operating in the async proxy.

**Async thread operating in generic proxy**. When an asynchronous operation is initiated, it is associated with an async thread, which is different from the CUDA thread that initiated the operation. _Preceding_ generic proxy (normal) loads and stores to the same address are guaranteed to be ordered before the asynchronous operation. However, _subsequent_ normal loads and stores to the same address are not guaranteed to maintain their ordering, potentially incurring a race condition until the async thread completes.

**Async thread operating in async proxy**. When an asynchronous operation is initiated, it is associated with an async thread, which is different from the CUDA thread that initiated the operation. _Prior and subsequent_ normal loads and stores to the same address are not guaranteed to maintain their ordering. A proxy fence is required to synchronize them across the different proxies to ensure proper memory ordering. Section [Using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-tma) demonstrates use of proxy fences to ensure correctness when performing asynchronous copies with TMA.

For more details on these concepts, see the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=proxy#proxies) documentation.
