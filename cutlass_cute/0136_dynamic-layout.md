---
title: "Dynamic Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html#dynamic-layout"
---

## [Dynamic Layout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#dynamic-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#dynamic-layout "Permalink to this headline")

In order to avoid generating and compiling multiple times for different shapes of the input `torch.Tensor`, CuTe DSL provides a way to
generate and compile JIT function with dynamic layout.

To get dyanmic layout of the `cute.Tensor`, a `torch.Tensor` object can be passed into the JIT function directly which instructs
CuTe DSL to call `cute.mark_layout_dynamic` automatically on the converted `cute.Tensor` per the leading dimension of the layout.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
```

In the example above, a single compilation of the JIT function `foo` is reused for different shapes of the input `torch.Tensor`.
This is possible because the converted `cute.Tensor` has a dynamic layout `(?,?):(?,1)` which is compatible with the shape of the
input `torch.Tensor` of both calls.

Alternatively, for compact layout, `cute.mark_compact_shape_dynamic` can be called for a finer-grained control to specify the mode
of the layout for dynamic and the divisibility constraint for the dynamic dimension.

Refer to [Integration with Frameworks](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html) for more details on `from_dlpack`, `mark_layout_dynamic`,
and `mark_compact_shape_dynamic`.
