---
title: "Dump the generated IR"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#dump-the-generated-ir"
---

### [Dump the generated IR](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#dump-the-generated-ir)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#dump-the-generated-ir "Permalink to this headline")

For users familiar with MLIR and compilers, CuTe DSL supports dumping the Intermediate Representation (IR).
This helps you verify whether the IR is generated as expected.

```bash
# Dump Generated CuTe IR (default: False)
export CUTE_DSL_PRINT_IR=1

# Keep Generated CuTe IR in a file (default: False)
export CUTE_DSL_KEEP_IR=1
```
