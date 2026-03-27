---
title: "C++ and Python Integration"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/index.html#c-and-python-integration"
---

## [C++ and Python Integration](https://docs.nvidia.com/cutlass/latest#c-and-python-integration)[](https://docs.nvidia.com/cutlass/latest/#c-and-python-integration "Permalink to this headline")

The C++ implementation modularizes the computational building blocks or “modular parts” into reusable software components using template classes. The Python DSLs, initially introduced as the CuTe DSL, provide an intuitive interface for rapid kernel development with fine-grained control over hardware behavior.

These abstractions allow developers to specialize and tune computation primitives for different layers of the parallelization hierarchy using configurable custom tiling sizes, data types, and other algorithmic policies through C++ template metaprogramming or dynamic Python APIs.
