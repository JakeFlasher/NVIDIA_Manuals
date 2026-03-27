---
title: "Working with torch Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#working-with-torch-tensors"
---

## [Working with torch Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#working-with-torch-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#working-with-torch-tensors "Permalink to this headline")

As you may have noticed in the examples above, TVM FFI-compiled functions can
directly accept `torch.Tensor` objects (and other DLPack-compatible tensors) as inputs.
The resulting functions add minimal overhead, enabling faster eager invocations
thanks to the optimized calling path.
