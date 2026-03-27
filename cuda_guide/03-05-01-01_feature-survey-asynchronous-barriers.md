---
title: "3.5.1.1. Asynchronous Barriers"
section: "3.5.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#feature-survey--asynchronous-barriers"
---

### [3.5.1.1. Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#asynchronous-barriers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#asynchronous-barriers "Permalink to this headline")

[Asynchronous barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers) were introduced in [Section 3.2.4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) and allow for more nuanced control over synchronization between threads. Asynchronous barriers separate the arrival and the wait of a barrier. This allows applications to perform work that does not depend on the barrier while waiting for other threads to arrive. Asynchronous barriers can be specified for different [thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-thread-scopes).  Full details of asynchronous barriers are found in [Section 4.9](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers).
