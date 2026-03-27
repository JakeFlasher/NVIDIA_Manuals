---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#dsl_control_flow--overview"
---

## [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#overview "Permalink to this headline")

CuTe DSL walks Python’s AST and converts each control-flow construct it finds into
structured intermediate representation (IR).  You can therefore write ordinary Python loops and branches
while the compiler decides—statement by statement—whether to

- **evaluate at compile time** if it’s a native Python control flow, or
- **emit intermediate representation (IR)** when the control flow is marked as dynamic.

Passing intermediate representation (IR) values to a native Python control flow will result in an error.

For a high-level discussion of the overall pipeline, see
[the code-generation overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html).
