---
title: "Set function name prefix"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#set-function-name-prefix"
---

### [Set function name prefix](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#set-function-name-prefix)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#set-function-name-prefix "Permalink to this headline")

By default, the function name (host function or kernel function) is automatically generated based on the function name and its parameters.
Sometimes you may want to attach some runtime information to the function name to make performance profiling and debugging easier,
e.g., the kernel configs or the rank ids. You can assign a name prefix to the name by calling the `set_name_prefix`
method on the  host function or kernel function.

```python
@cute.kernel
def kernel(arg1, arg2, ...):
    ...
@cute.jit
def launch_kernel():
    kernel.set_name_prefix("your_custom_name_prefix")
    kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
```

For above example, the generated kernel name will be “your_custom_name_prefix_xxx”.
