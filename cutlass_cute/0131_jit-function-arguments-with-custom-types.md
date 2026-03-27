---
title: "JIT function arguments with customized types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html#jit-function-arguments-with-custom-types"
---

## [JIT function arguments with customized types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#jit-function-arguments-with-custom-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#jit-function-arguments-with-custom-types "Permalink to this headline")

CuTe DSL supports customized types for JIT function arguments by providing two runtime checkable protocols:

- **`JitArgument` which is used for host JIT functions to be called from Python.**
  - `__c_pointers__`: Generate a list of ctypes pointers for the current object.
  - `__get_mlir_types__`: Generate a list of MLIR types for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.
- **`DynamicExpression` which is used for device JIT functions to be called from the host JIT functions.**
  - `__extract_mlir_values__`: Generate a dynamic expression for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.

Refer to [typing.py](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py) for more details on these protocol APIs.

Depending on different cases of the customized types, CuTe DSL provides easy ways to adopt customized types for JIT function arguments.
