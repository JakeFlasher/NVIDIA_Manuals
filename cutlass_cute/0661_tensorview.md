---
title: "TensorView"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#tensorview"
---

### [TensorView](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tensorview)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tensorview "Permalink to this headline")

Matrices and tensors used in linear algebra computations are invariably finite. `TensorView<class T, class Layout>` extends `TensorRef<>` by
adding an `extent` vector to describe the logical extent of the tensor or matrix.

Example:

```c++
int4_t *ptr = ...;
int ldm = ...;
MatrixCoord extent = ...;

int row = ...;
int column = ...;

layout::ColumnMajor layout(ldm);
TensorView<int4_t, layout::ColumnMajor> view(ptr, layout, extent);

MatrixCoord coord = {row, column};

if (view.contains(coord)) {     // verify coordinate is in bounds before performing access

  int4_t x = ref.at(coord);
  ref.at({row, column}) = x * 2_s4;
}
```

A `TensorView<>` may be constructed from a `TensorRef<>` succinctly as follows:

```c++
layout::ColumnMajor layout(ldm);
TensorRef<int4_t, layout::ColumnMajor> ref(ptr, layout);

TensorView<int4_t, layout::ColumnMajor> view(ref, extent);    // construct TensorView from TensorRef and extent
```

Note, computations avoid becoming overdetermined by accepting a single problem size component
and `TensorRef` objects for each of the operands whose extents are implied as a precondition of the operation. By avoiding
redundant storage of extent quantities, CUTLASS minimizes capacity utilization of precious resources such as constant memory.
This is consistent with BLAS conventions.
