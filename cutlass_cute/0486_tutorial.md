---
title: "Tutorial"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html#tutorial"
---

## [Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tutorial)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tutorial "Permalink to this headline")

This directory contains a CuTe tutorial in Markdown format.
The file
[`0x_gemm_tutorial.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
explains how to implement dense matrix-matrix multiply using CuTe components.
It gives a broad overview of CuTe and thus would be a good place to start.

Other files in this directory discuss specific parts of CuTe.

- [`01_layout.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) describes `Layout`, CuTe’s core abstraction.
- [`02_layout_algebra.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) describes more advanced `Layout` operations and the CuTe layout algebra.
- [`03_tensor.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html) describes `Tensor`,
a multidimensional array abstraction which composes `Layout`
with an array of data.
- [`04_algorithms.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html) summarizes CuTe’s
generic algorithms that operate on `Tensor`s.
- [`0t_mma_atom.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html) demonstrates CuTe’s meta-information and interface to our GPUs’
architecture-specific Matrix Multiply-Accumulate (MMA) instructions.
- [`0x_gemm_tutorial.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) walks through building a GEMM from scratch using CuTe.
- [`0y_predication.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html) explains what to do
if a tiling doesn’t fit evenly into a matrix.
- [`0z_tma_tensors.md`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html) explains an advanced `Tensor` type that CuTe uses to support TMA loads and stores.
