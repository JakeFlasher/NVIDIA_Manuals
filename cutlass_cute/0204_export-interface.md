---
title: "Export Interface"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#export-interface"
---

### [Export Interface](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#export-interface)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#export-interface "Permalink to this headline")

The `export_to_c` interface is provided by the `JitCompiledFunction` class. It accepts the following parameters:

- `file_path`: The path to the directory where the header and object files will be saved.
- `file_name`: The base name for the header and object files. The same file name will always overwrite existing files.
- `function_prefix`: The prefix of the function symbol in the generated object file. This should be a unique identifier to avoid symbol conflicts. Users should ensure the function prefix is unique for each exported function. Defaults to the `file_name`.

It generates the following files:

- `{file_path}/{file_name}.h`: A C header file containing API function declarations. This header specifies the runtime function signatures in C, mirroring the original Python function interfaces.
- `{file_path}/{file_name}.o`: A standard object file containing the compiled kernel code. You can link this object file into either a static or shared library. It includes the host entry function, fatbin data, and helper functions such as `cuda_init` and `cuda_load_to_device`. Additionally, it embeds metadata for runtime loading and version verification.

Example:

```python
import cutlass.cute as cute
import cutlass.cute.cuda as cuda

@cute.kernel
def print_tensor_kernel(a: cute.Tensor):
    cute.printf("a: {}", a)

@cute.jit
def print_tensor(a: cute.Tensor, stream: cuda.CUstream):
    print_tensor_kernel(a).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

compiled_func = cute.compile(print_tensor)
# Export compiled functions to object files and headers
compiled_func.export_to_c(file_path="./artifacts", file_name="print_tensor_example", function_prefix="print_tensor")
```
