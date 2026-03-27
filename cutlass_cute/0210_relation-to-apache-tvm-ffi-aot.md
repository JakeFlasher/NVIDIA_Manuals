---
title: "Relation to Apache TVM FFI AOT"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#relation-to-apache-tvm-ffi-aot"
---

## [Relation to Apache TVM FFI AOT](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#relation-to-apache-tvm-ffi-aot)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#relation-to-apache-tvm-ffi-aot "Permalink to this headline")

Apache TVM FFI AOT offers a comparable capability, enabling TVM functions to be compiled into binary files that can be loaded and executed at runtime.
For more information, see the section “Exporting Compiled Module” in [Compile with TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).

The primary distinction is that, when TVM FFI is enabled, CuTe DSL generates a dedicated wrapper function on top of the underlying CuTe ABI. This wrapper adheres to the calling conventions defined by TVM FFI.
In contrast, the CuTe ABI entry function is specified directly in the generated header file, which affects how arguments must be provided.

For instance, with the TVM FFI wrapper function, users are able to pass in arguments such as `torch.Tensor` directly. However, when calling the CuTe ABI entry function, arguments should be provided as `cute.Tensor` types.
