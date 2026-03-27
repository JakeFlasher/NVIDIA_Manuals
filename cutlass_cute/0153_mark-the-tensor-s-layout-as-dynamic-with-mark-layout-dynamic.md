---
title: "Mark the Tensor’s Layout as Dynamic with mark_layout_dynamic"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-layout-dynamic"
---

## [Mark the Tensor’s Layout as Dynamic with mark_layout_dynamic](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#mark-the-tensor-s-layout-as-dynamic-with-mark-layout-dynamic)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#mark-the-tensor-s-layout-as-dynamic-with-mark-layout-dynamic "Permalink to this headline")

After calling this function, all shape modes become dynamic. The stride modes also become dynamic
with the following two exceptions:

1. the leading dimension’s stride remains fixed at 1;
2. stride elements equal to 0 (which indicates broadcasting) are retained.

The full signature of `mark_layout_dynamic` is as follows:

```python
def mark_layout_dynamic(self, leading_dim: int|None = None):
```

The `leading_dim` parameter specifies the leading dimension of the tensor. The leading dimension’s
stride is set to 1 unless inconsistent with the layout of the DLPack tensor. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be 1,
the layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- If `leading_dim` is specified to be 0, a deduction failure error is raised because the stride of
dimension 0 is 2 (not 1).

The default value for `leading_dim` is `None`.  In such case, the system
automatically deduces it from the tensor’s layout using the following logic:

1. If exactly one dimension has stride 1, that dimension is the leading dimension.
2. If multiple dimensions have stride 1, deduction succeeds only when exactly one of them
has size > 1 (that dimension is used). If none or more than one has size > 1, an error is raised.
Note that after converting a **PyTorch** tensor to the DLPack format, the stride for dimensions
with size 1 are canonicalized to 1, which can produce multiple stride-1 dimensions.
3. If no dimension has stride 1, all strides remain dynamic.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the leading dimension is 1.
The layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- For a tensor with layout `(1,5,1):(1,1,1)`, multiple dimensions have stride 1 but exactly one
has size > 1 (dim 1). The leading dimension is deduced to be 1: `(?,?,?):(?,1,?)`.
- For a tensor with layout `(2,2):(8,2)`, no dimension has stride 1, so all strides remain
dynamic: `(?,?):(?,?)`.

The leading dimension accepts negative index which means the dimension is counted from the last dimension. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be -1,
the layout will be marked as `(?,?,?,?):(?,?,?,1)`.
