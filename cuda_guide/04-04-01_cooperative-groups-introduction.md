---
title: "4.4.1. Introduction"
section: "4.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups--introduction"
---

## [4.4.1. Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#introduction)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#introduction "Permalink to this headline")

Cooperative Groups are an extension to the CUDA programming model for organizing groups of collaborating threads.
Cooperative Groups allow developers to control the granularity at which threads are collaborating, helping them to express richer, more efficient parallel decompositions.
Cooperative Groups also provide implementations of common parallel primitives like scan and parallel reduce.

Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the `__syncthreads()` intrinsic function.
In an effort to express broader patterns of parallel interaction, many performance-oriented programmers have resorted to writing their own ad hoc and unsafe primitives for synchronizing threads within a single warp, or across sets of thread blocks running on a single GPU.
Whilst the performance improvements achieved have often been valuable, this has resulted in an ever-growing collection of brittle code that is expensive to write, tune, and maintain over time and across GPU generations.
Cooperative Groups provides a safe and future-proof mechanism for writing performant code.

The full Cooperative Groups API is available in the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-partition-header).
