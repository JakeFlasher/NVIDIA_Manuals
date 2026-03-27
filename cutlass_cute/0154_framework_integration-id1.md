---
title: "Code Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#framework_integration--id1"
---

### [Code Example](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#id1)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#id1 "Permalink to this headline")

The following example demonstrates how to use `mark_layout_dynamic` to specify dynamic tensor layouts.

- `t0` shows the usage of `mark_layout_dynamic` with unspecified `leading_dim` and the automatic deduction of leading dimension.
- `t1` & `t2` shows the usage of `mark_layout_dynamic` with specified `leading_dim`.
- `t3` shows the usage of `mark_layout_dynamic` with no leading dimension.
- `t4` shows the usage of `mark_layout_dynamic` with broadcasted dimensions.
- `t5` shows automatic deduction for tensor `b` (multiple stride-1, exactly one has size > 1 → dim 1).
- `t5_fail` demonstrates the deduction failure when multiple dimensions have stride 1 but none has size > 1.
- `t6` & `t7` demonstrate incorrect settings for `leading_dim` and expected errors.

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)
# (2,2):(8,2)
c = torch.empty(3, 4)[::2, ::2]
# (3,1,1,5):(5,0,0,1)
d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)

# auto deduce the leading dimension to be 3
t0 = from_dlpack(a).mark_layout_dynamic()
print(t0)
# (?,?,?,?):(?,?,?,1)

t1 = from_dlpack(b).mark_layout_dynamic(leading_dim=0)
print(t2)
# (?,?,?,?,?):(1,?,?,?,?)

t2 = from_dlpack(b).mark_layout_dynamic(leading_dim=2)
print(t3)
# (?,?,?,?,?):(?,?,1,?,?)

t3 = from_dlpack(c).mark_layout_dynamic()
print(t3)
# (?,?):(?,?)

t4 = from_dlpack(d).mark_layout_dynamic()
print(t4)
# (?,?,?,?):(?,0,0,1)

# b has layout (1,4,1,32,1):(1,1,1,4,1); dim 1 has size > 1, so deduction succeeds to dim 1.
t5 = from_dlpack(b).mark_layout_dynamic()
print(t5)
# (?,?,?,?,?):(?{i64},1,?{i64},?{i64},?{i64})

# Rejected: multiple stride-1, none with size > 1 (e.g. torch.ones(1,1,1)).
t5_fail = from_dlpack(torch.ones(1, 1, 1)).mark_layout_dynamic()
# Can't deduce the leading dimension from layout (multiple dimensions have stride 1 but none has size > 1)...

t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
# Expected strides[leading_dim] == 1, but got 16

t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
# Expected strides[leading_dim] == 1, but got 4

c = torch.empty(1000000000, 1000000000)
t8 = from_dlpack(c, use_32bit_stride=True).mark_layout_dynamic()
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```
