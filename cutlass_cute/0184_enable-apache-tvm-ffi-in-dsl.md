---
title: "Enable Apache TVM FFI in CuTe DSL"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#enable-apache-tvm-ffi-in-dsl"
---

## [Enable Apache TVM FFI in CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#enable-apache-tvm-ffi-in-dsl)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#enable-apache-tvm-ffi-in-dsl "Permalink to this headline")

First, install the `tvm-ffi` package by following its [installation guide](https://tvm.apache.org/ffi/#installation).

There are two ways to enable TVM FFI in CuTe DSL:

1. Use the `options` argument in `cute.compile` to specify the TVM FFI option. For example:

```python
# Assuming you have defined a function `add` decorated with @cute.jit
def example_compile():
   a_torch = torch.randn(10, 20, 30).to(torch.float16)
   b_torch = torch.randn(10, 20, 30).to(torch.float16)
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add = cute.compile(add, a_torch, b_torch, options="--enable-tvm-ffi")
```

Note that the object returned by `cute.compile` is a Python function specific to TVM FFI.

2. Alternatively, you can enable TVM FFI globally by setting the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. Please note that this setting will apply to all JIT compilations within the environment.
