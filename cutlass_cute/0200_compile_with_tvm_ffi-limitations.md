---
title: "Limitations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#compile_with_tvm_ffi--limitations"
---

## [Limitations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#limitations)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#limitations "Permalink to this headline")

The Fake Tensor flow is ONLY compatible with TVM FFI because TVM FFI supports more flexible constraints on Tensor arguments.
For instance, fake tensor can specify per-mode static shape or constraints on shape and strides which are not supported by
existing `from_dlpack` flow. It’s expected that JIT function compiled with fake tensor will have different ABI compared to
tensor converted by `from_dlpack`.

```python
import cutlass.cute as cute
import torch

n = cute.sym_int()
# Dynamic Shape
fake_a = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

# Compile without tvm-ffi
compiled_fn = cute.compile(foo, fake_a)

# Wrong, in compatible ABI
compiled_fn(from_dlpack(a))
```

In order to avoid such issue, it’s recommended to use fake tensor only with TVM FFI backend. Practically speaking,
as we only want to call `from_dlpack` once and reuse for both compilation and runtime, the benefit of
using fake tensor is limited in this case.
