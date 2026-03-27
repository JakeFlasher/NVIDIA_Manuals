---
title: "Dump the generated PTX & CUBIN"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#dump-the-generated-ptx-cubin"
---

### [Dump the generated PTX & CUBIN](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#dump-the-generated-ptx-cubin)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#dump-the-generated-ptx-cubin "Permalink to this headline")

For users familiar with PTX and SASS, CuTe DSL supports dumping the generated PTX and CUBIN.

```bash
# Dump generated PTX in a .ptx file (default: False)
export CUTE_DSL_KEEP_PTX=1

# Dump generated cubin in a .cubin file (default: False)
export CUTE_DSL_KEEP_CUBIN=1
```

To further get SASS from cubin, users can use `nvdisasm` (usually installed with CUDA toolkit) to disassemble the cubin.

```bash
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```
