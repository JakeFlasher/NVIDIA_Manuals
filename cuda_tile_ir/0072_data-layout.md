---
title: "Data Layout"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#data-layout"
---

#### [Data Layout](https://docs.nvidia.com/cuda/tile-ir/latest/sections#data-layout)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#data-layout "Permalink to this headline")

Allocations pointed to by input pointer values, and by extension views (see [below](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-views)), must conform
to the specified data layout.

We expect that the allocation pointed to by *ptr<E>* is a sized contiguous allocation of scalar values of element type *E*.

There is no padding between elements of the allocation, and we expect that for an allocation of size *N*
will be equivalent to *N * sizeof(E)* bytes. The size and encoding of a element is determined by its type
*E* and is defined in the table below.

As an aside the datatype encoding is compatible with [DLPack](https://github.com/dmlc/dlpack) a standard
adopted by most deep learning frameworks and array libraries. We provide the equivalent
PyTorch and NumPy encodings for each datatype.

For NumPy low-precision types we provide the equivalent in terms of the
[ml_dtypes](https://github.com/jax-ml/ml_dtypes) library a standard collection of
low-precision NumPy data types.

> **Warning**
>
> **Tile IR** layouts are currently restricted to be contiguous for sub-byte types.

| **Tile IR** Type | DLPack Type Code | DLPack Bits | DLPack Lanes | NumPy Type | PyTorch Type |
| --- | --- | --- | --- | --- | --- |
| i1 | `kDLInt`, `kDLUInt` | 8 | 1 | `numpy.uint8` ([unpacked](https://numpy.org/devdocs/reference/generated/numpy.packbits.html)) | N/A |
| i8 | `kDLInt`, `kDLUInt` | 8 | 1 | `numpy.uint8`, `numpy.int8` | `torch.bool` |
| i16 | `kDLInt`, `kDLUInt` | 16 | 1 | `numpy.int16`, `numpy.uint16` | `torch.int16`, `torch.uint16` |
| i32 | `kDLInt`, `kDLUInt` | 32 | 1 | `numpy.int32`, `numpy.uint32` | `torch.int32`, `torch.uint32` |
| i64 | `kDLInt`, `kDLUInt` | 64 | 1 | `numpy.int64`, `numpy.uint64` | `torch.int64`, `torch.uint64` |
| f16 | `kDLFloat` | 16 | 1 | `numpy.float16` | `torch.float16` |
| f32 | `kDLFloat` | 32 | 1 | `numpy.float32` | `torch.float32` |
| f64 | `kDLFloat` | 64 | 1 | `numpy.float64` | `torch.float64` |
| bf16 | `kDLBfloat` | 16 | 1 | `ml_dtypes.bfloat16` | `torch.bfloat16` |
| fp8 (E4M3) | `kDLFloat8_e4m3` | 8 | 1 | `ml_dtypes.float8_e4m3fn` | `torch.float8_e4m3fn` |
| fp8 (E5M2) | `kDLFloat8_e5m2` | 8 | 1 | `ml_dtypes.float8_e5m2` | `torch.float8_e5m2` |

> **Note**
>
> Allocations are the only values where memory layout is specified in **Tile IR**.
