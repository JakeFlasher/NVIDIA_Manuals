---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html#heuristics--overview"
---

## [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#overview "Permalink to this headline")

Gemm heuristics in `cutlass_library` aim to reduce the search space for runtime autotuning, so that only a subset of valid kernels need to be built and profiled for a given set of GEMM problems. This implementation uses Nvidia’s `nvidia-matmul-heuristics`, an analytical heuristic that ranks GEMM kernels by estimated performance given a problem size and hardware SKU. You can find more info in [the docs](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics).
