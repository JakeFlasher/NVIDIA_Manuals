---
title: "Leveraging TVM FFI for Faster PyTorch Interop"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#leveraging-tvm-ffi-for-faster-pytorch-interop"
---

## [Leveraging TVM FFI for Faster PyTorch Interop](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#leveraging-tvm-ffi-for-faster-pytorch-interop)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#leveraging-tvm-ffi-for-faster-pytorch-interop "Permalink to this headline")

The latest version of CuTe DSL supports TVM FFI to improve interoperability with PyTorch
and other machine learning frameworks. Using TVM FFI provides the following features:

- Faster JIT function invocation.
- Direct acceptance of `torch.Tensor` objects as function arguments.
- Enhanced error handling and kernel validation.
- Seamless integration with multiple programming languages.

For more details, see [Compile with TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).
