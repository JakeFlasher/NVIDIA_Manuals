---
title: "cute.Tensor adapter for TVM FFI"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#cute-tensor-adapter-for-tvm-ffi"
---

## [cute.Tensor adapter for TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#cute-tensor-adapter-for-tvm-ffi)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#cute-tensor-adapter-for-tvm-ffi "Permalink to this headline")

To adapt the `cute.Tensor` to the TVM FFI function, you can use the `cute.runtime.from_dlpack` function with the
`enable_tvm_ffi=True` option or the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. For example:

```python
def example_from_dlpack():
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add_one(a_cute, b_cute)
```

Note that because the `cute.runtime.from_dlpack` function performs an explicit DLPack conversion, it is less efficient than passing the `torch.Tensor` directly.
You can also use `cute.Tensor` as an argument hint for `cute.compile`.

```python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```
