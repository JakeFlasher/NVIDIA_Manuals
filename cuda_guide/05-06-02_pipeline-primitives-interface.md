---
title: "5.6.2. Pipeline Primitives Interface"
section: "5.6.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#pipeline-primitives-interface"
---

## [5.6.2. Pipeline Primitives Interface](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#pipeline-primitives-interface)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#pipeline-primitives-interface "Permalink to this headline")

Pipeline primitives provide a C-like interface for the functionality available in `<cuda/pipeline>`. The pipeline primitives interface is available by including the `<cuda_pipeline.h>` header. When compiling without ISO C++ 2011 compatibility, include the `<cuda_pipeline_primitives.h>` header.

> **Note**
>
> The pipeline primitives API only supports tracking asynchronous copies from global memory to shared memory with specific size and alignment requirements. It provides equivalent functionality to a `cuda::pipeline` object with `cuda::thread_scope_thread`.
