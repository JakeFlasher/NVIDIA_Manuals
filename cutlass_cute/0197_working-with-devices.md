---
title: "Working with Devices"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#working-with-devices"
---

## [Working with Devices](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#working-with-devices)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#working-with-devices "Permalink to this headline")

TVM FFI-compiled functions naturally work across GPU devices.
The device index of the first input GPU tensor determines the kernel’s device context.
The TVM FFI function calls `cudaSetDevice` to set the correct device
before launching the kernel based on that tensor’s device index.
For advanced scenarios that pass raw pointers instead of tensors, you should call
`cudaSetDevice` explicitly through the CUDA Python API.
