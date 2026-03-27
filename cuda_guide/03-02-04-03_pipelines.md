---
title: "3.2.4.3. Pipelines"
section: "3.2.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#pipelines"
---

### [3.2.4.3. Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#pipelines)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#pipelines "Permalink to this headline")

The CUDA programming model provides the pipeline synchronization object as a coordination mechanism to sequence asynchronous memory copies into multiple stages, facilitating the implementation of double- or multi-buffering producer-consumer patterns. A pipeline is a double-ended queue with a _head_ and a _tail_ that processes work in a first-in first-out (FIFO) order. Producer threads commit work to the pipeline’s head, while consumer threads pull work from the pipeline’s tail.

Pipelines are exposed through the `cuda::pipeline` API in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html) library, as well as through a [primitives API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#pipeline-primitives-interface). The following tables describe the main functionality of the two APIs.

| `cuda::pipeline` API | Description |
| --- | --- |
| `producer_acquire` | Acquires an available stage in the pipeline’s internal queue. |
| `producer_commit` | Commits the asynchronous operations issued after the `producer_acquire` call on the currently acquired stage of the pipeline. |
| `consumer_wait` | Waits for completion of asynchronous operations in the oldest stage of the pipeline. |
| `consumer_release` | Releases the oldest stage of the pipeline to the pipeline object for reuse. The released stage can be then acquired by a producer. |

| Primitives API | Description |
| --- | --- |
| `__pipeline_memcpy_async` | Request a memory copy from global to shared memory to be submitted for asynchronous evaluation. |
| `__pipeline_commit` | Commits the asynchronous operations issued before the call on the current stage of the pipeline. |
| `__pipeline_wait_prior(N)` | Waits for completion of asynchronous operations in all but the last N commits to the pipeline. |

The `cuda::pipeline` API has a richer interface with less restrictions, while the primitives API only supports tracking asynchronous copies from global memory to shared memory with specific size and alignment requirements. The primitives API provides equivalent functionality to a `cuda::pipeline` object with `cuda::thread_scope_thread`.

For detailed usage patterns and examples, see [Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines).
