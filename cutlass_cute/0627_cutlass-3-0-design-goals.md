---
title: "CUTLASS 3.0 design goals"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html#cutlass-3-0-design-goals"
---

## [CUTLASS 3.0 design goals](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-3-0-design-goals)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-3-0-design-goals "Permalink to this headline")

CUTLASS 3.0 has the following design goals, in no particular order.

- Simplify expressing and manipulating data and thread layouts across
the GEMM hierarchy with CuTe layouts and layout algebra.
- Improve code readability and learning curve by
reducing the number of named types.
- Functional correctness by default,
actionable static asserts otherwise.
- Single, clear points of performance tuning and custom kernel extensions.
- Support for NVIDIA Hopper GPUs with great performance using
features such as Tensor Cores, tensor memory accelerator, and thread block clusters.
