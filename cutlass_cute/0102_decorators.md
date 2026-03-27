---
title: "Decorators"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#decorators"
---

## [Decorators](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#decorators)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#decorators "Permalink to this headline")

CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1. `@jit` — Host-side JIT-compiled functions
2. `@kernel` — GPU kernel functions

Both decorators can optionally use a **preprocessor** that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying IR.
