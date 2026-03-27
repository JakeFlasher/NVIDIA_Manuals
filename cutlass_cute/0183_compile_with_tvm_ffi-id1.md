---
title: "Compile with TVM FFI"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#compile_with_tvm_ffi--id1"
---

# [Compile with TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#id1)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#id1 "Permalink to this headline")

Apache TVM FFI is an open ABI and FFI for machine learning systems. More information can be found in
the [official documentation](https://tvm.apache.org/ffi/).

To install TVM FFI, you can run the following command:

```bash
pip install apache-tvm-ffi
# optional package for improved torch tensor calling performance
pip install torch-c-dlpack-ext
```

In CuTe DSL, TVM FFI can be enabled as an option for JIT-compiled functions. Using TVM FFI can lead to faster
JIT function invocation and provides better interoperability with machine learning frameworks
(e.g., directly take `torch.Tensor` as arguments).
