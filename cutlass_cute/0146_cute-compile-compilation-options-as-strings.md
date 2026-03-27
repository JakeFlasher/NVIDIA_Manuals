---
title: "cute.compile Compilation Options as strings"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html#cute-compile-compilation-options-as-strings"
---

## [cute.compile Compilation Options as strings](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#cute-compile-compilation-options-as-strings)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#cute-compile-compilation-options-as-strings "Permalink to this headline")

You can provide additional compilation options as a string when calling `cute.compile`. The CuTe DSL uses `argparse` to parse these options and will raise an error if any invalid options are specified.

| **Option** | **Description** | **Default** | **Type** |
| --- | --- | --- | --- |
| `opt-level` | Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is [0, 3]. | 3 (highest level of optimization) | int |
| `enable-assertions` | Enable host and device code assertions. | False | bool |
| `keep-cubin` | Keep the generated CUBIN file. | False | bool |
| `keep-ptx` | Keep the generated PTX file. | False | bool |
| `ptxas-options` | The options to pass to the PTX Compiler library. | “” | str |
| `generate-line-info` | Generate line information for debugging. | False | bool |
| `gpu-arch` | The GPU architecture to compile for. | “” | str |
| `enable-tvm-ffi` | Enable Apache TVM FFI. | False | bool |

You can use the following code to specify compilation options:

```python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```
