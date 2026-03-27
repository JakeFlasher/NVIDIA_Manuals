---
title: "3. Meta-Programming vs Runtime: Two Worlds in One Function"
section: "3"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#meta-programming-vs-runtime-two-worlds-in-one-function"
---

## [3. Meta-Programming vs Runtime: Two Worlds in One Function](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#meta-programming-vs-runtime-two-worlds-in-one-function)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#meta-programming-vs-runtime-two-worlds-in-one-function "Permalink to this headline")

A key insight for understanding CuTe DSL is that **your Python code runs twice**,
in two very different contexts:

1. **Meta-programming time (compilation)** — Python executes to _build_ the
kernel.  This happens on the host CPU when you call a `@jit` function.
2. **Runtime (execution)** — The compiled kernel runs on the GPU with actual
tensor data.

This distinction determines what you can observe and when.
