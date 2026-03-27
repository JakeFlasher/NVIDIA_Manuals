---
title: "CUTLASS Layout Concept"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#cutlass-layout-concept"
---

## [CUTLASS Layout Concept](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-layout-concept)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-layout-concept "Permalink to this headline")

CUTLASS Layouts are a systematic design pattern for the following:

- Mapping _logical_ index space to _physical_ offsets in memory
- Storing the dynamic state needed in the above computation
- Defining a type system for partial specialization of other CUTLASS components

_Concept:_ layouts satisfy the following concept.

```c++
/// CUTLASS Layout concept example
struct LayoutConcept {

  /// Logical rank of tensor
  static int const kRank;

  /// Rank of stride vector
  static int const kStrideRank;

  /// Index type used for coordinates
  struct Index;

  /// Long index type used for offsets
  struct LongIndex;

  /// Logical coordinate - satisfies Coord<kRank, ..>
  struct TensorCoord;

  /// Stride object - satisfies Coord<kStrideRank, ..>
  struct Stride

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  LayoutConcept();

  /// Ctor
  CUTLASS_HOST_DEVICE
  LayoutConcept(Stride stride);

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static LayoutConcept packed(TensorCoord const &extent);

  /// Function call operator returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (row, column)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const;

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const;

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const;

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride();

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const;
};
```

_Layout_ objects generalize leading dimensions of matrices typical in _BLAS_ implementations. For example, cuBLAS assumes
Fortran-style _column-major_ layouts of matrices and refers to this as the matrix’s “leading dimension.”

```c++
cublasGemmEx(
  ...
  ptr_A,      // pointer to first element of matrix A
  lda,        // leading dimension
  ...
);
```

This implies an element at coordinate (_row_, _column_) has offset `row + lda * column`.

This is equivalently represented by CUTLASS’s `layout::ColumnMajor` type as follows.

```c++
layout::ColumnMajor layout(lda);

int offset = layout({row, column});     // returns row  + lda * column
```

Other layout functions are possible such as row-major:

```c++
layout::RowMajor layout(lda);

int offset = layout({row, column});     // returns lda * row + column
```

In both cases, the _logical_ coordinate (_row_, _column_) is represented by the same object. This enables an algorithm to be
implemented as generic template, with locations within tensors always specified in logical space. _Layout_ objects map this to
physical offsets in memory.

The layout’s `::packed()` static method may be used to construct a layout object given the extent of a densely packed tensor.
This method is needed when an algorithm must define a buffer of arbitrary layout.

Example:

```c++
typename ArbitraryLayout::TensorCoord extent = make_Coord(...);
typename ArbitraryLayout::TensorCoord coord;

ArbitraryLayout layout = ArbitraryLayout::packed(extent);

int offset = layout({coord});
```

The layout’s `::capacity()` method computes the number of locations in memory needed to represent a tensor. This is
useful when allocating memory, as more storage may be needed than what is strictly necessary for a fully packed
tensor.

Example:

```c++
int lda = columns + padding;
MatrixCoord extent{rows, columns};

layout::RowMajor layout(lda);

auto capacity = layout.capacity(extent);    // returns rows * (columns + padding)
```
