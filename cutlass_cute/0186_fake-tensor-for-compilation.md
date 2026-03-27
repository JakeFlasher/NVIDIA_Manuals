---
title: "Fake tensor for compilation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#fake-tensor-for-compilation"
---

## [Fake tensor for compilation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#fake-tensor-for-compilation)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#fake-tensor-for-compilation "Permalink to this headline")

The TVM FFI function accepts DLPack-compatible tensors as arguments, such as those from torch or jax.
However, during compilation, it is necessary to specify the tensors’ dynamic properties in CuTe DSL.
To clearly distinguish between the compilation phase and runtime,
CuTe DSL provides a “fake tensor” that can be used for compilation. For example:

```python
import cutlass.cute as cute
import torch

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
   )

def example_add_one():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # now compiled_add_one is a TVM-FFI function that can be called with torch.Tensor as input
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(a_torch, b_torch)
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

The fake tensor is a placeholder that mimics the interface of a real tensor but does not hold real data or allow indexing.
It is used in compilation or testing scenarios where only shape/type/layout information is needed.
All attempts to access or mutate data will raise errors.
