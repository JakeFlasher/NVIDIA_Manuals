---
title: "Using Python’s print and CuTe’s cute.printf"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#using-python-s-print-and-cute-s-cute-printf"
---

### [Using Python’s print and CuTe’s cute.printf](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#using-python-s-print-and-cute-s-cute-printf)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#using-python-s-print-and-cute-s-cute-printf "Permalink to this headline")

CuTe DSL programs can use both Python’s native `print()` as well as our own `cute.printf()`  to
print debug information during kernel generation and execution. They differ in a few key ways:

- Python’s `print()` executes during compile-time only (no effect on the generated kernel) and is
typically used for printing static values (e.g. a fully static layouts).
- `cute.printf()` executes at runtime on the GPU itself and changes the PTX being generated. This
can be used for printing values of tensors at runtime for diagnostics, but comes at a performance
overhead similar to that of *printf()* in CUDA C.

For detailed examples of using these functions for debugging, please refer to the associated
notebook referenced in [Educational Notebooks](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/notebooks.html).
