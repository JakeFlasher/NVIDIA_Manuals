---
title: "CuTe MMA Atoms"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#cute-mma-atoms"
---

## [CuTe MMA Atoms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-mma-atoms)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-mma-atoms "Permalink to this headline")

CuTe exposes each MMA to generic CUDA C++ code as a pair of structs:
an “Operation” struct,
and an `MMA_Traits` struct templated on the Operation struct type.

An “Operation” struct exposes the PTX instruction
for that specific operation.
It defines the arguments and interface it expects.
Operation structs have minimal software dependencies –
they do not use layouts, tensors, or non-standard numeric data types – and
describe only the physical inputs and outputs to the instruction.
Different structs have different names
that describe what the MMA instruction does.
We will explain the naming scheme below.

A corresponding `MMA_Traits` struct specialization
defines meta-information about the Operation,
such as the logical compute types, the logical shape of the operation,
and the `Layout`s of threads and values within the operation.
The `MMA_Traits` struct takes the Operation as a template parameter.
CuTe specializes `MMA_Traits` for each Operation type that it supports.

Together, these two types comprise an “Atom” that decouples the complexity of thread and data layouts from the call site of the PTX instruction.  The Atom’s Traits struct exposes information that is relevant to a single MMA operation, no matter the granularity at which it operates.

CuTe MMA atoms expose the semantics of a single MMA operation.
This is true regardless of the hardware level at which the MMA operates.
CuTe supports MMA atoms that operate at a variety of hardware levels,
including

- a single thread (e.g., fused multiply-add (FMA) instruction);
- a quadpair (Volta);
- a single warp (Ampere); and
- a warpgroup (Hopper).
