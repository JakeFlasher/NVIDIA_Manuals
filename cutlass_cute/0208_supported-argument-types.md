---
title: "Supported Argument Types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#supported-argument-types"
---

## [Supported Argument Types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#supported-argument-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#supported-argument-types "Permalink to this headline")

CuTe DSL supports the following argument types:

- `cute.Tensor`
- `cute.Shape` / `cute.Coord` / `cute.Tile` / `cute.IntTuple` / `cute.Stride`
- `cuda.CUstream`
- `cutlass.Int8` / `cutlass.Int16` / `cutlass.Int32` / `cutlass.Int64` / `cutlass.Boolean`
- `cutlass.Uint8` / `cutlass.Uint16` / `cutlass.Uint32` / `cutlass.Uint64`
- `cutlass.Float32` / `cutlass.TFloat32` / `cutlass.Float64` / `cutlass.Float16`

Note that:

1. `cute.Tensor` is a dynamic tensor type that only contains dynamic shapes and strides in its ABI representation. As a result, different compilations may produce different tensor ABIs. This is why declarations for each tensor type are included in the generated header file.
2. `strides` in `cute.Tensor` are determined by the `use_32bit_strides` compile argument. When `use_32bit_strides` is set to `True`, the strides are 32-bit; when set to `False`, they are 64-bit.
3. Currently, custom types are not supported for AOT compilation.
