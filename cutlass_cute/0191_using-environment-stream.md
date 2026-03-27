---
title: "Using Environment Stream"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#using-environment-stream"
---

### [Using Environment Stream](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#using-environment-stream)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#using-environment-stream "Permalink to this headline")

The second option is to rely on the environment stream flag.
Pass `use_tvm_ffi_env_stream=True` to `make_fake_stream` to mark the stream argument as an
environment stream, which means it no longer needs to be provided explicitly.
TVM FFI will automatically use its environment stream (i.e., the current PyTorch stream)
as the stream argument. The example below demonstrates this flow:

```python
def example_add_one_with_env_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   # we will use TVM FFI environment stream
   stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   with torch.cuda.stream(torch_stream):
      # no need to pass in the stream explicitly, env stream will be synced
      # to torch.cuda.current_stream() before the function call.
      compiled_add_one(a_torch, b_torch)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

Using the environment stream flag both speeds up calls and simplifies integration
with frameworks such as PyTorch, since no explicit stream parameter is required.
We recommend using the environment stream flag to both simplify framework integration
and minimize host-side calling overhead.
