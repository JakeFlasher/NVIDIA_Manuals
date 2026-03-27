---
title: "Implicit Conversion"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#implicit-conversion"
---

## [Implicit Conversion](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#implicit-conversion)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#implicit-conversion "Permalink to this headline")

Tensors originating from frameworks supporting the DLPack protocol can be directly provided to a
JIT function as a regular parameter. CuTe DSL’s  runtime implicitly converts the original tensor to a
CuTe tensor with a fully dynamic layout except for the stride element corresponding to the leading
dimension. The example below demonstrates this use case.

```python
import torch
import cutlass.cute as cute

@cute.jit
def foo(src):
    """
    The following lines print

    ptr<f32, generic> o (?,?,?):(?,?,1)
    <class 'cutlass.cute.core._Tensor'>
    """
    print(src)
    print(type(src))

a = torch.randn(30, 20, 32, device="cpu")
foo(a)
```
