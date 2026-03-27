---
title: "Static Layout vs. Dynamic Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html#static-layout-vs-dynamic-layout"
---

## [Static Layout vs. Dynamic Layout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#static-layout-vs-dynamic-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#static-layout-vs-dynamic-layout "Permalink to this headline")

Per the previous sections, we have seen that static layout leads to distinct JIT code generations while dynamic layout leads to a single
compilation for different shapes.

That said, creating JIT function with static layout is useful when the use cases targeting input data with fixed shapes.
Since more information is available at compile time, the compiler would be able to kick in optimizations that otherwise would not
be possible for the code generated for dynamic layout.

On the other hand, dynamic layout would be more flexible for the cases where the input data has varying shapes. This provides more
scalability of the generated code to deal with varying input data of different shapes.
