---
title: "4.11.1. Using LDGSTS"
section: "4.11.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-ldgsts"
---

## [4.11.1. Using LDGSTS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-ldgsts)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-ldgsts "Permalink to this headline")

Many CUDA applications require frequent data movement between global and shared memory. Often, this involves copying smaller data elements or performing irregular memory access patterns. The primary goal of LDGSTS (CC 8.0+, see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)) is to provide an efficient asynchronous data transfer mechanism from global memory to shared memory for smaller, element-wise data transfers while enabling better utilization of compute resources through overlapped execution.

**Dimensions**. LDGSTS supports copying 4, 8, or 16 bytes. Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode, in which case data is also cached in the L1, while copying 16-bytes enables the L1 BYPASS mode, in which case the L1 is not polluted.

**Source and destination**. The only direction supported for asynchronous copy operations with LDGSTS is from global to shared memory. The pointers need to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied. Best performance is achieved when the alignment of both shared memory and global memory is 128 bytes.

**Asynchronicity**. Data transfers using LDGSTS are [asynchronous](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async thread operations (see [Async Thread and Async Proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. _Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future_.

LDGSTS must provide a signal when the operation is complete. LDGSTS can use [shared memory barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) or  [pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines) as mechanisms to provide completion signals. By default, each thread only waits for its own LDGSTS copies. Thus, if you use LDGSTS to prefetch some data that will be shared with other threads, a `__syncthreads()` is necessary after synchronizing with the LDGSTS completion mechanism.

| Source | Destination | Completion Mechanism | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global |  |  |
| global | shared::cta | shared memory barrier, pipeline | [cuda::memcpy_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html), [cooperative_groups::memcpy_async](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-async-memcpy), [__pipeline_memcpy_async](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#pipeline-primitives-interface) |
| global | shared::cluster |  |  |
| shared::cluster | shared::cta |  |  |
| shared::cta | shared::cta |  |  |

In the following sections, we will demonstrate how to use LDGSTS through examples and explain the differences between the different APIs.
