---
title: "Mark the Tensor’s Layout as Dynamic with mark_compact_shape_dynamic"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-compact-shape-dynamic"
---

## [Mark the Tensor’s Layout as Dynamic with mark_compact_shape_dynamic](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#mark-the-tensor-s-layout-as-dynamic-with-mark-compact-shape-dynamic)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#mark-the-tensor-s-layout-as-dynamic-with-mark-compact-shape-dynamic "Permalink to this headline")

The `mark_compact_shape_dynamic` function provides fine-grain control over dynamic shapes for compact
layouts. The full signature of `mark_compact_shape_dynamic` is as follows:

```python
def mark_compact_shape_dynamic(self, mode: int, stride_order: tuple[int, ...]|None = None, divisibility: int = 1):
```

The `mode` parameter determines which shape dimension becomes dynamic. After calling this function,
the specific shape dimension given by `mode` is marked as dynamic immediately. The stride will be
updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.

The `stride_order` parameter specifies the ordering of strides in the tensor. It is consistent
with `torch.Tensor.dim_order()` and defaults to `None`. The parameter indicates the order of
modes (dimensions) if the current layout were to be converted to row-major order. It starts from the
outermost to the innermost dimension when reading it from left to right. This parameter must be
explicitly set when the stride order cannot be automatically deduced from the tensor’s layout, such
as when multiple dimensions have a stride of 1.

For example:

- Layout `(4,2):(1,4)` has a `stride_order` of `(1,0)` indicates the innermost dimension is
0 (`4:1`), the outermost dimension is 1 (`2:4`).
- Layout `(5,3,2,4):(3,1,15,30)` has a `stride_order` of `(3,2,0,1)` indicates the innermost
dimension is 1 (`3:1`), the outermost dimension is 3 (`4:30`).

If `stride_order` is not specified, the system automatically deduces it from the tensor’s layout
using the following logic:

1. Sort the strides in descending order.
2. If multiple dimensions have a stride of 1, a deduction failure error is raised.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the deduced `stride_order` is `[3,2,0,1]`.
- For a tensor with layout `(1,5,1):(1,1,1)`, `stride_order`’s deduction fails because
all dimensions have an identical stride of 1, making it impossible to determine the correct ordering.

If `stride_order` is specified, the system validates that the order is consistent with the
tensor’s layout.

The `divisibility` parameter specifies the divisibility of the dynamic shape. It could be used to
represent the assumption alignment of the input. Defaults to 1.

Note that this API is only available for compact tensors. For non-compact tensors, we can use
`cute.assume` to attach divisibility information to a specific shape mode in a host JIT function,
as demonstrated in the following example:

```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    # use cute.assume to set shape of mode=0 with divisibility=16
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```
