---
title: "4.10.3. Consuming Work"
section: "4.10.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#consuming-work"
---

## [4.10.3. Consuming Work](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#consuming-work)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#consuming-work "Permalink to this headline")

Consuming work from a previously committed stage involves:

> - Collectively waiting for the stage to complete, e.g., using `pipeline.consumer_wait()` to wait on the tail (oldest) stage, from a set of consumer threads.
> - Collectively _releasing_ the stage using `pipeline.consumer_release()`.

With `cuda::pipeline<cuda:thread_scope_thread>` one can also use the `cuda::pipeline_consumer_wait_prior<N>()` friend function to wait for all except the last N stages to complete, similar to `__pipeline_wait_prior(N)` in the primitives API.
