---
title: "Correctness by default, Performance through clear, individual points of tuning"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html#correctness-by-default-performance-through-clear-individual-points-of-tuning"
---

## [Correctness by default, Performance through clear, individual points of tuning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#correctness-by-default-performance-through-clear-individual-points-of-tuning)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#correctness-by-default-performance-through-clear-individual-points-of-tuning "Permalink to this headline")

CUTLASS 2.x maintained its thread layouts as implicit indexing math implemented
as a part of 1D iterators. This meant that the thread to data layout mapping
was implicit in the imperative structure of the C++ code itself and did not have
a formal algebra we could use to manipulate these mappings. Each iterator
had to re-implement its indexing and mapping logic. This made it hard to learn
how this mapping was performed for existing iterators, and even harder to
implement custom layout functions for the core inner loops of a GEMM.

CUTLASS 3.0 replaces all iterator concepts from CUTLASS 2.x
with a single layout type for thread and data tensors.
CuTe’s formalized layout algebra is then used at every layer of
the GEMM hierarchy to manipulate the mapping between the two.
CuTe layouts always maintain logical consistency, and for fully static layouts
(such as in the core unrolled inner loops), provide
compile time checks that break builds if this consistency is violated.
In this way, CuTe reifies the thread-to-data-layout mapping,
makes it easier to write code that is “correct by construction”.
If the code compiles, it’s probably correct.
