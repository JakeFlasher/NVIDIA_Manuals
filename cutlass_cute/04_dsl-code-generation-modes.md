---
title: "4. CuTe DSL Code-Generation Modes"
section: "4"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#dsl-code-generation-modes"
---

## [4. CuTe DSL Code-Generation Modes](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#dsl-code-generation-modes)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#dsl-code-generation-modes "Permalink to this headline")

CuTe’s Python front-end combines the techniques above into **two mutually
exclusive modes** (see [Left: tracing mode records only the path that executed.
Right: preprocessor mode emits structured intermediate representation (IR) for every branch and loop
before tracing the arithmetic.](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#fig-dsl-modes)), selectable with the `preprocessor` flag of the
`@jit` decorator:

1. Tracing mode `@jit(preprocess=False)` – tracing only.
This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be
straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.

2.  Preprocessor mode (**default**) `@jit(preprocess=True)` – **AST rewrite + tracing**.
The AST pass captures every loop and branch, eliminating the correctness and
optimisation problems of pure tracing; tracing then fills in the arithmetic.
This hybrid “preprocessor” pipeline is unique to CuTe DSL and was designed
specifically to overcome the disadvantages identified above.
