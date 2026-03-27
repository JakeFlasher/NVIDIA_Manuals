---
title: "CuTe’s support for Matrix Multiply-Accumulate instructions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#cute-s-support-for-matrix-multiply-accumulate-instructions"
---

# [CuTe’s support for Matrix Multiply-Accumulate instructions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-s-support-for-matrix-multiply-accumulate-instructions)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-s-support-for-matrix-multiply-accumulate-instructions "Permalink to this headline")

In this file, we explain in detail how we support our GPUs’
Matrix Multiply-Accumulate (MMA) hardware instructions in CuTe.

MMAs are architecture-specific.
Different generations of GPU architectures
introduce different sets of MMA instructions.
However, CuTe features such as `Layout`
makes it possible to expose MMAs for use in generic CUDA C++ code.
We accomplish this in multiple steps.

1. We wrap each MMA’s PTX instruction in an “Operation” struct.
2. For each Operation struct, we define a “Traits” struct
that defines all of the meta-information needed to use the Operation.
3. Combining the above, an “Atom” is the combination of the PTX Operation struct with the
meta-information Traits struct and provides methods to construct
`cute::Tensor` “fragments” for that Operation and to use that Operation
on existing `cute::Tensor`s.
4. Combining potentially multiple Atoms, a “TiledMMA” provides utilities for building
more complex partitioning patterns by creating layouts and interleavings of Atoms.
