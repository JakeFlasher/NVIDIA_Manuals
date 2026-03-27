---
title: "Exporting Compiled Module"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#exporting-compiled-module"
---

## [Exporting Compiled Module](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#exporting-compiled-module)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#exporting-compiled-module "Permalink to this headline")

The TVM FFI function supports exporting the compiled module to an object file
for further use. For example:

```python
import subprocess
import cutlass.cute as cute

def example_add_one_export():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_one.export_to_c("./add_one.o", function_name="add_one")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_one.so", "./add_one.o", *runtime_libs]
   print(cmd)
   subprocess.run(cmd, check=True)
   print(f"Successfully created shared library: ./add_one.so")
```

Then you can load back the exported module and use it in different ways:

```python
import torch
from cutlass import cute

def example_load_module_add_one():
   mod = cute.runtime.load_module("./add_one.so", enable_tvm_ffi=True)
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   mod.add_one(a_torch, b_torch)
   print("result of b_torch after mod.add_one(a_torch, b_torch)")
   print(b_torch)
```

The exported object file exposes the function symbol `__tvm_ffi_add_one` that is
compatible with TVM FFI and can be used in various frameworks and programming languages.
You can either build a shared library and load it back, or link the object file directly
into your application and invoke the function via the `InvokeExternC` mechanism in TVM FFI.
For more information, see the [quick start guide](https://tvm.apache.org/ffi/get_started/quickstart)
in the official documentation.

When you build your own libraries, make sure you link against the necessary runtime libraries.
You can use `cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)` to get the path to these libraries.
`cute.runtime.load_module(path, enable_tvm_ffi=True)` will load these libraries automatically before loading
an exported module. You can also manually load these libraries in advanced use cases.

For low-level cute ABI AOT compilation support without TVM FFI, you can refer to [Ahead-of-Time (AOT) Compilation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html).
