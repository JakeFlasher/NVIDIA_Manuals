---
title: "Summary of Control Flow behavior"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#summary-of-control-flow-behavior"
---

## [Summary of Control Flow behavior](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#summary-of-control-flow-behavior)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#summary-of-control-flow-behavior "Permalink to this headline")

| **Control Flow** | **Run time evaluation** | **Compile time evaluation** |
| --- | --- | --- |
| if cutlass.const_expr() | ❌ | ✅ |
| if pred | ✅ | ❌ |
| while cutlass.const_expr() | ❌ | ✅ |
| while pred | ✅ | ❌ |
| for i in cutlass.range_constexpr() | ❌ | ✅ |
| for i in range() | ✅ | ❌ |
| for i in cutlass.range() (support advanced unrolling and pipelining) | ✅ | ❌ |
