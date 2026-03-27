---
title: "4.8.1. Background"
section: "4.8.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management--background"
---

## [4.8.1. Background](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#background)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#background "Permalink to this headline")

Traditionally, the only indication of a failed CUDA API call is the return of a non-zero code.
As of CUDA Toolkit 12.9, the CUDA Runtime defines over 100 different return codes
for error conditions, but many of them are generic and give the developer no assistance with debugging the cause.
