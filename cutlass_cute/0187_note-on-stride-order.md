---
title: "Note on Stride Order"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#note-on-stride-order"
---

### [Note on Stride Order](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#note-on-stride-order)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#note-on-stride-order "Permalink to this headline")

Note that CuTe’s convention is to write the stride order for dimensions from left to right,
where a lower order number means higher priority. In the context of the `make_fake_compact_tensor` API,
for shape `(2, 3, 4)` and stride order `(0, 1, 2)`, the stride is `(1, 2, 6)`.
This is commonly known as column-major order. If you want to create a fake tensor with compact row-major order,
you should explicitly pass in `stride_order=tuple(reversed(range(len(shape))))`
to `make_fake_compact_tensor`. Alternatively, you can always precisely control the
stride via the `stride` argument in the `make_fake_tensor` API.
