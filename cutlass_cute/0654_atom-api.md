---
title: "Atom API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#atom-api"
---

## [Atom API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#atom-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#atom-api "Permalink to this headline")

An “Atom” is the smallest collection of threads and data
that must participate in the execution of a hardware-accelerated
math or copy operation.

An Atom is “atomic” (indivisible) not in the sense of
concurrent memory operations like `atomicAdd`
(which are “indivisible in time (causality)”),
but in the sense of indivisibility in “space” –
the number of values and the groups of parallel workers
that must participate in the operation together.

An Atom uses CuTe Layouts to express the required
dimensions and strides of its input and output arrays.
Generally these are fixed at compile time.

The Atom API wraps calls to actual hardware instructions
that accelerate MMA or copy operations.
Users can ask for GPU architecture-specific implementations,
or just pick generic implementations and rely on
whatever GPU architectures were enabled.

For more information about Atoms,
please refer to CuTe’s tutorial, e.g., the sections on

- [algorithms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html) like `gemm` and `copy`,
- [MMA Atoms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#cute-mma-atoms), and
- [a GEMM example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html).
