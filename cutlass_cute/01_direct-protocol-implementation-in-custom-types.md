---
title: "1. Direct protocol implementation in customized types"
section: "1"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html#direct-protocol-implementation-in-custom-types"
---

### [1. Direct protocol implementation in customized types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#direct-protocol-implementation-in-custom-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#direct-protocol-implementation-in-custom-types "Permalink to this headline")

One way is to implement the protocol methods directly in the customized types to enable the protocol based JIT function argument generation.

```python
import cutlass
import cutlass.cute as cute

# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor # Dynamic argument
        self._offset = offset # Dynamic argument

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```

In the example above, the `MyDynamicExpression` implements the `DynamicExpression` protocol and CuTe DSL will generate the JIT function arguments for the JIT kernel `my_kernel` based on the protocol methods.
