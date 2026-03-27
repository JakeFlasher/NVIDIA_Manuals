---
title: "cpasync submodule"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#module-cutlass.cute.nvgpu.cpasync"
---

# [cpasync submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.nvgpu.cpasync)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.nvgpu.cpasync "Permalink to this headline")

```
_`class`_`cutlass.cute.nvgpu.cpasync.``LoadCacheMode`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.LoadCacheMode "Link to this definition")
```

Bases: `Enum`

An enumeration for the possible cache modes of a non-bulk `cp.async` instruction.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyG2SOp`(
```

Bases: `CopyOp`

Non-bulk asynchronous GMEM to SMEM Copy Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy).

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyBulkTensorTileG2SOp`(
```

Bases: `TmaCopyOp`

Bulk tensor asynchrnous GMEM to SMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")__`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp.cta_group "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyBulkTensorTileG2SMulticastOp`(
```

Bases: `TmaCopyOp`

Bulk tensor asynchrnous multicast GMEM to SMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

```
`cta_group`_`:` [`CtaGroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")__`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp.cta_group "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyBulkTensorTileS2GOp`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp "Link to this definition")
```

Bases: `TmaCopyOp`

Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

```
`__init__`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp.__init__ "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyReduceBulkTensorTileS2GOp`(
```

Bases: `TmaCopyOp`

Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk).
This Operation uses TMA in the `.tile` mode.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.cpasync.``CopyDsmemStoreOp`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp "Link to this definition")
```

Bases: `CopyOp`

Asynchronous Store operation to DSMEM with explicit synchronization.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async).

```
`__init__`() → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp.__init__ "Link to this definition")
```

```
`cutlass.cute.nvgpu.cpasync.``make_tiled_tma_atom`(
```

Makes a TMA Copy Atom in the `.tile` mode to copy tiles of a GMEM tensor to/from SMEM
buffer with the given Layout.

Given

- a GMEM tensor
- a SMEM layout
- a CTA-level Tiler

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
layout while maintaining consistency with the provided Tiler.

This function returns two results:

1. the Copy Atom
2. a TMA tensor that maps logical coordinates of the GMEM tensor to coordinates consumed by the        TMA unit. TMA tensors contain basis stride elements that enable their associated layout to        compute coordinates. Like other CuTe tensors, TMA tensors can be partitioned.

**Parameters:**
: - **op** (_TMAOp_) – The TMA Copy Operation to construct an Atom
- **gmem_tensor** (_Tensor_) – The GMEM tensor involved in the Copy
- **smem_layout** (_Union__[__Layout__,__ComposedLayout__]_) – The SMEM layout to construct the Copy Atom, either w/ or w/o the stage mode
- **cta_tiler** (_Tiler_) – The CTA Tiler to use
- **num_multicast** (_int_) – The multicast factor
- **internal_type** (_Type__[__Numeric__]_) – Optional internal data type to use when the tensor data type is not supported by the TMA unit

**Returns:**
: A TMA Copy Atom associated with the TMA tensor

**Return type:**
: Tuple[[atom.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

```
`cutlass.cute.nvgpu.cpasync.``tma_partition`(
```

Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.

```
`cutlass.cute.nvgpu.cpasync.``create_tma_multicast_mask`(
```

Computes a multicast mask for a TMA load Copy.

**Parameters:**
: - **cta_layout_vmnk** (_Layout_) – The VMNK layout of the cluster
- **cta_coord_vmnk** (_Coord_) – The VMNK coordinate of the current CTA
- **mcast_mode** (_int_) – The tensor mode in which to multicast

**Returns:**
: The resulting mask

**Return type:**
: Int16

```
`cutlass.cute.nvgpu.cpasync.``prefetch_descriptor`(
```

Prefetches the TMA descriptor associated with the TMA Atom.

```
`cutlass.cute.nvgpu.cpasync.``copy_tensormap`(
```

Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
pointer.

**Parameters:**
: - **tma_atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – The TMA Copy Atom
- **tensormap_ptr** (_Pointer_) – The pointer to the memory location to copy the tensormap to

```
`cutlass.cute.nvgpu.cpasync.``update_tma_descriptor`(
```

Updates the TMA descriptor in the memory location pointed to by the provided pointer using
information from a TMA Copy Atom and the provided GMEM tensor.

Specifically, the following fields of the TMA descriptor will be updated:

1. the GMEM tensor base address
2. the GMEM tensor shape
3. the GMEM tensor stride

Other fields of the TMA descriptor are left unchanged.

**Parameters:**
: - **tma_atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – The TMA Copy Atom
- **gmem_tensor** (_Tensor_) – The GMEM tensor
- **tensormap_ptr** (_Pointer_) – The pointer to the memory location of the descriptor to udpate

```
`cutlass.cute.nvgpu.cpasync.``fence_tma_desc_acquire`(
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

```
`cutlass.cute.nvgpu.cpasync.``cp_fence_tma_desc_release`(
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy).

```
`cutlass.cute.nvgpu.cpasync.``fence_tma_desc_release`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.cpasync.fence_tma_desc_release "Link to this definition")
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).
