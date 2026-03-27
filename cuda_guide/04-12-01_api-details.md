---
title: "4.12.1. API Details"
section: "4.12.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#api-details"
---

## [4.12.1. API Details](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#api-details)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#api-details "Permalink to this headline")

Cancelling a thread block via the cluster launch control API is done
asynchronously and synchronized using a shared memory barrier,
following a programming pattern similar
to [asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies).

The API, available through [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html), provides:

- A request instruction that writes encoded cancellation results to a `__shared__` variable.
- Decoding instructions that extract success/failure status and the cancelled thread block index.

Note that cluster launch control operations are modeled as async proxy operations (see [Async Thread and Async Proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)).
