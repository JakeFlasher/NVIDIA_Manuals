---
title: "Common"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_common.html#module-cutlass.cute.nvgpu"
---

# [Common](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.nvgpu)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.nvgpu "Permalink to this headline")

```
_`class`_`cutlass.cute.nvgpu.``OpError`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.OpError "Link to this definition")
```

Bases: `DSLBaseError`

An exception class for Op construction errors.

```
`cutlass.cute.nvgpu.``normalize_field_to_ir_name`(_`field`_, _`admissible_fields`_) → `str`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.normalize_field_to_ir_name "Link to this definition")
```

Normalize a field specifier to its IR logical field name.

Accepted inputs:

- Enum value present in admissible_fields (must expose _to_ir_field_name()).
- Exact string IR name (e.g., “accum_c”, “neg_a”, “sf_a”).

Any other form is rejected.

```
_`class`_`cutlass.cute.nvgpu.``MmaUniversalOp`(_`abacc_dtype``:` `Type``[``cutlass.cute.typing.Numeric``]`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.MmaUniversalOp "Link to this definition")
```

Bases: `MmaOp`

The universal MMA Operation.

This Operation currently expects the A/B operands as well as the accumulator to share the same
data types.

**Parameters:**
: **abacc_dtype** (_Type__[__Numeric__]_) – The data type for the A/B operands and the accumulator

```
`abacc_dtype`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.MmaUniversalOp.abacc_dtype "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.``MmaUniversalTrait`(_`value``:` `cutlass._mlir.ir.Value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.MmaUniversalTrait "Link to this definition")
```

Bases: `Trait`

```
_`class`_`cutlass.cute.nvgpu.``CopyUniversalOp`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.CopyUniversalOp "Link to this definition")
```

Bases: `CopyOp`

The universal Copy Operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL
)
```

- `tensor_dtype` is the data type used to build the reference TV Layout (either the source         or the destination TV Layout) in unit of tensor elements and is used for partitioning by         `TiledCopy` for example
- `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom         execution. This can be larger than the width of the above data type. When not provided,         the compiler will do a best effort at auto-vectorizing.
- `l1c_evict_priority` is a kw argument specifying the L1 cache eviction priority hint for         the copy operation. Defaults to `EVICT_NORMAL` if not provided.
- `invariant` is a kw argument specifying whether the load is invariant (read-only data         that never changes). This enables compiler optimizations like instruction reordering.         Defaults to `False` if not provided.

```
_`class`_`cutlass.cute.nvgpu.``CopyUniversalTrait`(_`value``:` `cutlass._mlir.ir.Value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.CopyUniversalTrait "Link to this definition")
```

Bases: `Trait`

```
_`class`_`cutlass.cute.nvgpu.``MemoryOrder`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.MemoryOrder "Link to this definition")
```

Bases: `Enum`

An enumeration.

```
_`class`_`cutlass.cute.nvgpu.``MemoryScope`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.MemoryScope "Link to this definition")
```

Bases: `Enum`

An enumeration.

```
_`class`_`cutlass.cute.nvgpu.``CacheEvictionPriority`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.CacheEvictionPriority "Link to this definition")
```

Bases: `Enum`

An enumeration.

```
`cutlass.cute.nvgpu.``make_tiled_tma_atom_A`(
```

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the MK projections of the TiledMMA for A tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled_mma (considering the M-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
similarly to any other CuTe tensors using the algebra.

**Parameters:**
: - **op** (_Union__[_[_CopyBulkTensorTileG2SOp_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")_,_[_CopyBulkTensorTileG2SMulticastOp_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")_]_) – The Copy Operation to construct an Atom for
- **gmem_tensor** (_Tensor_) – The GMEM tensor to be loaded by this copy atom
- **smem_layout** (_Union__[__Layout__,__ComposedLayout__]_) – Shared memory layout to load the tensor into (PDSL)
- **mma_tiler_mnk** (_Shape_) – The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
- **tiled_mma** ([_atom.TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")) – The TiledMMA that will consume the load as operands
- **cluster_shape_vmnk** (_Shape_) – The Cluster-level shape in VMNK dimensions
- **internal_type** (_Type__[__Numeric__]_) – An optional parameter for the internal data type to when element
type does not match the copy type

**Returns:**
: A copy atom for this operation and the associated TMA coord tensor

**Return type:**
: Tuple[[atom.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

```
`cutlass.cute.nvgpu.``make_tiled_tma_atom_B`(
```

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the NK projections of the TiledMMA for B tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled_mma (considering the N-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
similarly to any other CuTe tensors using the algebra.

**Parameters:**
: - **op** (_Union__[_[_CopyBulkTensorTileG2SOp_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")_,_[_CopyBulkTensorTileG2SMulticastOp_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")_]_) – The Copy Operation to construct an Atom for
- **gmem_tensor** (_Tensor_) – The GMEM tensor to be loaded by this copy atom
- **smem_layout** (_Union__[__Layout__,__ComposedLayout__]_) – Shared memory layout to load the tensor into (PDSL)
- **mma_tiler_mnk** (_Shape_) – The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
- **tiled_mma** (_core.TiledMma_) – The TiledMMA that will consume the load as operands
- **cluster_shape_vmnk** (_Shape_) – The Cluster-level shape in VMNK dimensions
- **internal_type** (_Type__[__Numeric__]_) – An optional parameter for the internal data type to when element
type does not match the copy type

**Returns:**
: A Copy Atom for this Operation and the associated TMA tensor

**Return type:**
: Tuple[[atom.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]
