---
title: "Programming with Static and Dynamic Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html#programming-with-static-and-dynamic-layout"
---

## [Programming with Static and Dynamic Layout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#programming-with-static-and-dynamic-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#programming-with-static-and-dynamic-layout "Permalink to this headline")

CuTe DSL provides intuitive way to program with static and dynamic layout in the codes.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

In this example, the JIT function `foo` is compiled with a static layout `(3):(1)` for the first call, which means the
size of the tensor is known at compile time. CuTe DSL makes good use of this and automatically handles the if condition at the
compile time. Hence the generated codes are efficient without the if condition at all.

For the second call, the JIT function `foo` is compiled with a dynamic layout `(?):(1)` hence the tensor size is only
evaluated at runtime. CuTe DSL automatically generates the code to handle the dynamic layout and the if condition at runtime.

The same applies to loop as well:

```python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

With the static layout in the first call, CuTe DSL is able to fully unroll the loop at compile time. While in the second call,
the generated codes will have the loop executed at runtime based on the dynamic layout.

With the single JIT function implementation, CuTe DSL is able to handle control-flow constructs and automatically generate
the optimized codes for different cases. This is all possible because CuTe DSL is able to walk the Python AST and convert
each control-flow construct it finds accordingly.

Please refer to [Control Flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html) for more details.
