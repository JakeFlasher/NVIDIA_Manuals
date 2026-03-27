---
title: "1.1 AST Rewrite"
section: "1.1"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#ast-rewrite"
---

### [1.1 AST Rewrite](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#ast-rewrite)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#ast-rewrite "Permalink to this headline")

The function’s abstract-syntax tree is analysed **before** execution.
Python control-flow (`for`/`while`, `if`/`else`) and built-ins are converted to structured intermediate representation (IR)
constructs.  Computation inside each region is left untouched at this stage.

_Advantages_

- Sees the entire program, so every branch and loop is preserved.
- Keeps loop structure intact for optimization such as tiling, vectorisation
or GPU thread mapping.

_Disadvantages_

- Requires a well-defined Python subset that the rewriter understands.
