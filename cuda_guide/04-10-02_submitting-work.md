---
title: "4.10.2. Submitting Work"
section: "4.10.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#submitting-work"
---

## [4.10.2. Submitting Work](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#submitting-work)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#submitting-work "Permalink to this headline")

Committing work to a pipeline stage involves:

> - Collectively _acquiring_ the pipeline _head_ from a set of producer threads using `pipeline.producer_acquire()`.
> - Submitting asynchronous operations, e.g., `memcpy_async`, to the pipeline head.
> - Collectively _committing_ (advancing) the pipeline head using `pipeline.producer_commit()`.

If all resources are in use, `pipeline.producer_acquire()` blocks producer threads until the resources of the next pipeline stage are released by consumer threads.
