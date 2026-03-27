---
title: "1.3 The Hybrid Solution"
section: "1.3"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#the-hybrid-solution"
---

### [1.3 The Hybrid Solution](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#the-hybrid-solution)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#the-hybrid-solution "Permalink to this headline")

As shown above, neither technique alone is sufficient—but together they
complement each other perfectly.

**Why this works: GPU kernels are simple at runtime**

High-performance GPU kernels are structurally simple at runtime: they avoid
deep call hierarchies, complex branching, and dynamic dispatch.  However,
_authoring_ such kernels benefits greatly from Python’s abstractions—classes,
metaprogramming, and polymorphic patterns improve readability and
maintainability.

The hybrid approach resolves this tension by evaluating Python abstractions at
compile time while emitting simple, optimized code for runtime execution.

**How |DSL| divides the work:**

1. **AST rewrite handles structure** — loops (`for`, `while`) and branches
(`if`/`else`) are converted to structured intermediate representation (IR) _before_ execution.
This solves tracing’s control-flow problem.
2. **Tracing handles arithmetic** — inside each structured region, the tracer
records tensor operations exactly as they execute.  No need to model Python’s
complex semantics—just run Python and record what happens.  This solves AST
rewriting’s complexity problem.

The result:

- Loops compile to real loops, not unrolled traces.
- All branches are preserved, even if not taken during tracing.
- Dynamic shapes, metaprogramming, and Python idioms work naturally.
- The rewriter only needs to understand control flow, not all of Python.
