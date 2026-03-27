---
title: "1. Hybrid DSL: Python Metaprogramming, Structured GPU Code"
section: "1"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#hybrid-dsl-python-metaprogramming-structured-gpu-code"
---

## [1. Hybrid DSL: Python Metaprogramming, Structured GPU Code](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#hybrid-dsl-python-metaprogramming-structured-gpu-code)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#hybrid-dsl-python-metaprogramming-structured-gpu-code "Permalink to this headline")

CuTe DSL is a **hybrid DSL** that combines two compilation techniques: _AST rewrite_
and _tracing_.  This combination gives you the best of both worlds:

- **Program structure is preserved** — control flow (loops, branches) is
captured via AST rewrite, compiling to proper structured code instead of
flattened traces.
- **Python stays Python** — arithmetic and tensor operations are captured via
tracing, so dynamic shapes, metaprogramming, and Python’s rich expression
language work naturally.

To understand why this matters, let’s look at each technique.
