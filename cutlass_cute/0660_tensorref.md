---
title: "TensorRef"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#tensorref"
---

### [TensorRef](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tensorref)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tensorref "Permalink to this headline")

`TensorRef<class T, class Layout>` is a structure containing both a pointer to the start of a
tensor and a layout object to access its elements. This is a convenient object which may be
passed to functions to limit an explosion of arguments when the number of stride elements is
numerous.

Example:

```c++
int4_t *ptr = ...;
int ldm = ...;

int row = ...;
int column = ...;

layout::ColumnMajor layout(ldm);
TensorRef<int4_t, layout::ColumnMajor> ref(ptr, layout);

int4_t x = ref.at({row, column});     // loads a 4-bit signed integer from the tensor

ref.at({row, column}) = x * 2_s4;     // transforms this quantity and stores it back
```
