---
title: "Code Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#code-example"
---

### [Code Example](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#code-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#code-example "Permalink to this headline")

The following code demonstrates how to convert a PyTorch tensor to a CuTe tensor using the
`from_dlpack` function with default parameters.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)
```

Once converted, we can access the tensor’s information through various
attributes. The following list shows the attributes of the converted tensor:

- `tensor.shape`: the tensor’s shape
- `tensor.stride`: the tensor’s stride
- `tensor.memspace`: the tensor’s memory space
- `tensor.element_type`: the tensor’s element data type

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)

print(y.shape)        # (30, 20)
print(y.stride)       # (20, 1)
print(y.memspace)     # generic (if torch tensor in on device memory, memspace will be gmem)
print(y.element_type) # Float32
print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
```

The string format of the resulting CuTe tensor is

```console
Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>
```

As can be seen in the example above, `from_dlpack` first results in a tensor with a static layout.
To obtain dynamic or mixed static/dynamic layouts after calling `from_dlpack`, the
`mark_layout_dynamic` and `mark_compact_shape_dynamic` functions are used and described in
the following sections.
