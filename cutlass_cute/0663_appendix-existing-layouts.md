---
title: "Appendix: Existing Layouts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#appendix-existing-layouts"
---

### [Appendix: Existing Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#appendix-existing-layouts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#appendix-existing-layouts "Permalink to this headline")

This section enumerates several existing Layout types defined in CUTLASS.

Matrix layouts:

- `PitchLinear`: data layout defined by _contiguous_ and _strided_ dimensions. _contiguous_ refers to consecutive elements in memory, where as _strided_ refers to data separated by a uniform stride
– Rank: 2
– TensorCoord type: `PitchLinearCoord`
– Shape type: `PitchLinearShape`
– Stride rank: 1
- `ColumnMajor`: data layout defined by _rows_ and _columns_ dimensions. Can be mapped to `PitchLinear` by: (_contiguous_ = _rows_, _strided_ = _columns_)
– Rank: 2
– TensorCoord type: `MatrixCoord`
– Shape type: `MatrixShape`
– Stride rank: 1
- `RowMajor`: data layout defined by _rows_ and _columns_ dimensions. Can be mapped to `PitchLinear` by: (_contiguous_ = _columns_, _strided_ = _rows_)
– Rank: 2
– TensorCoord type: `MatrixCoord`
– Shape type: `MatrixShape`
– Stride rank: 1
- `ColumnMajorInterleaved<k>`: data layout defined by _rows_ and _columns_ dimensions. Data is packed into a ‘column-major’ arrangement of row vectors of fixed length.
– Rank: 2
– TensorCoord type: `MatrixCoord`
– Shape type: `MatrixShape`
– Stride rank: 1
- `RowMajorInterleaved<k>`: data layout defined by _rows_ and _columns_ dimensions. Data is packed into a ‘row-major’ arrangement of column vectors of fixed length.
– Rank: 2
– TensorCoord type: `MatrixCoord`
– Shape type: `MatrixShape`
– Stride rank: 1

Tensor layouts:

- `TensorNHWC`:

Permuted Shared Memory Layouts:

- `TensorOpCongruous<ElementSize>`
- `TensorOpCrosswise<ElementSize>`
