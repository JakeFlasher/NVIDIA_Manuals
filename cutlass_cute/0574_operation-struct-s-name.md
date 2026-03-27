---
title: "Operation struct’s name"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#operation-struct-s-name"
---

#### [Operation struct’s name](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#operation-struct-s-name)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#operation-struct-s-name "Permalink to this headline")

A CuTe Operation struct’s name principally encodes the PTX instruction it wraps.
These often include

- its first supported architecture,
- the M, N, and K dimensions that it accepts,
- the types that it takes, and
- the arrangement of the A and B inputs.

For example, the Volta section below will refer to the
`SM70_8x8x4_F32F16F16F32_NT` Operation struct defined in
[`include/cute/arch/mma_sm70.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/mma_sm70.hpp).

- “SM70” refers to Volta.
- “8x8x4” refers to M = 8, N = 8, and K = 4,
the dimensions of the MMA operation that the quadpair performs
(see below). This is reflected in the PTX as `.m8n8k4.`.
- “F32F16F16F32” refers to the element types
of the four matrix operands A, B, C, and D.
An MMA computes D = C + A * B,
so we read the types from left to right:
D is F32 (`float`), A is F16 (half),
B is F16 (half), and C is F32 (`float`). This is reflected in the PTX instruction name as `.f32.f16.f16.f32`.
- “NT” means that the PTX instruction is designed for inputs A as M-major (not transposed, column-major)
and inputs B as N-major (transposed, row-major). This is reflected in the PTX instruction name as `.col.row.`.
