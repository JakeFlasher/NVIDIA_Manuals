---
title: "Source Code Correlation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#source-code-correlation"
---

## [Source Code Correlation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#source-code-correlation)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#source-code-correlation "Permalink to this headline")

CuTe DSL provides Python code to PTX/SASS correlation to enable the profiling/debugging of generated kernels with debug symbols by generating line info when compiling the kernel.

You can enable that globally via the environment variable CUTE_DSL_LINEINFO=1. Alternative, you can use compilation options to enable that per kernel. Please refer to [JIT Compilation Options](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html) for more details.
