---
title: "5.3.2. Tensor View"
section: "5.3.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#tensor-view"
---

### [5.3.2. Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tensor-view)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tensor-view "Permalink to this headline")

It is common that data in global memory follows a strided structure. For
example, the widely adopted row-major or column-major layouts are
strided. It is beneficial for the compiler to be aware of the strided
layout of data in memory. **Tile IR** features a tensor view type to
describe such structure in global memory.

Conceptually, a tensor view type describes an abstracted tensor of
pointers. Like a regular tensor, it is described by a shape and the type
of elements it points to. It in addition has a vector of striding
factors.

The striding factors describe the relative position of locations the
elements of the tensor view point to. If an element is \(\(d\)\)
elements apart in dimension \(\(i\)\) of the tensor view, the
corresponding locations in memory will be \(\(d * stride_i\)\) elements
away. This information can be used by the compiler to reason about
access patterns and layouts of data in memory.

Values of type tensor view are typically never materialized in memory.
Rather, they are stored as a compact description of a base-location,
shape and striding factors. From this information, a tensor of pointers
corresponding to the full view value can be computed using the following
formula

$$
\[elem_{[i_0, ..., i_n]} = baseptr + \sum_{m=0}^{n} i_m * s_m\]
$$

where the \(\(s_m\)\) are the striding factors of the tensor view and
\(\(baseptr\)\) is the start address in global memory.

Other than the tile type, a tensor view supports dynamic extents in the
shape and stride vectors. We use the `?` character to denote these. Dynamic
extents are bound at runtime to values when the view type is
constructed using a [cuda_tile.make_tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-make-tensor-view) operation.

A tensor view cannot be directly used to access memory. It first needs to be
divided into tiles of static size. See [Subview Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-sec-subview-types) for
options to do so.

A tensor view is constructed using the
[cuda_tile.make_tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-make-tensor-view) operation and consists of the
components:

- `element_type`, the type of the elements in the view
- `shape`, an array of 64-bit integer describing the size of each
dimension
- `strides`, an array of 64-bit integer describing the stride of each
dimension
