---
title: "4.11. Asynchronous Data Copies"
section: "4.11"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies--asynchronous-data-copies"
---

# [4.11. Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#asynchronous-data-copies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#asynchronous-data-copies "Permalink to this headline")

Building on [Section 3.2.5](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies), this section provides detailed guidance and examples for asynchronous data movement within the GPU memory hierarchy. It covers LDGSTS for element-wise copies, the Tensor Memory Accelerator (TMA) for bulk (one-dimensional and multi-dimensional) transfers, and STAS for register to distributed shared memory copies, and shows how these mechanisms integrate with [asynchronous barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers) and [pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines).
