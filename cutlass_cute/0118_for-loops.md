---
title: "For Loops"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#for-loops"
---

## [For Loops](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#for-loops)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#for-loops "Permalink to this headline")

CuTe DSL recognises three kinds of ranges for `for` loops:

- `range` – the Python built-in, always lowered to intermediate representation (IR)
- `cutlass.range` - Same as Python built-in `range`, but supports advanced unrolling and pipelining control
- `cutlass.range_constexpr` – unrolled at compile time
