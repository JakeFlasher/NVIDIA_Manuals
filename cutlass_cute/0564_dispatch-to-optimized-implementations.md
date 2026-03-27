---
title: "Dispatch to optimized implementations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#dispatch-to-optimized-implementations"
---

### [Dispatch to optimized implementations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#dispatch-to-optimized-implementations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#dispatch-to-optimized-implementations "Permalink to this headline")

Just like with `copy`, CuTe’s implementations of `gemm`
uses its `Tensor` arguments’ types to dispatch
to an appropriately optimized implementation.
Also like `copy`, `gemm` takes an optional `MMA_Atom` parameter
that lets callers override the default `FMA` instruction
that CuTe would select based on the `Tensor` arguments’ types.

For more information on `MMA_Atom` and on specialization of `gemm`
for different architectures, please refer to the
[MMA section of the tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html).
