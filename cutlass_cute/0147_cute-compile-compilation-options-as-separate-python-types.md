---
title: "cute.compile Compilation Options as separate Python types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html#cute-compile-compilation-options-as-separate-python-types"
---

## [cute.compile Compilation Options as separate Python types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#cute-compile-compilation-options-as-separate-python-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#cute-compile-compilation-options-as-separate-python-types "Permalink to this headline")

Alternatively, you can also use a more Pythonic way to specify compilation options with separate Python types.
Compilation options can be programmatically composed using tuple and passed to `cute.compile` separately.

```python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

This approach causes invalid options to raise errors immediately, making it much easier to detect typos when specifying multiple options.
Notebly, boolean options are automatically converted to True instances of the option type for convenience.

```python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```
