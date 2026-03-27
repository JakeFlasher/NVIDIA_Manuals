---
title: "Utilities for SM100"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#module-cutlass.utils.sm100"
---

# [Utilities for SM100](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.utils.sm100)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.utils.sm100 "Permalink to this headline")

```
`cutlass.utils.sm100.``compute_epilogue_tile_shape`(
```

Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

**Parameters:**
: - **cta_tile_shape** (_cute.Shape_) – A tuple or list representing the dimensions of the CTA tile, where
cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
corresponds to the width (N) of the tile.
- **use_2cta_instrs** (_bool_) – A flag indicating whether the configuration is for a 2SM setup.
- **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
- **elem_ty_d** (_Type__[__Numeric__]_) – The element type of output tensor D.
- **layout_c** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")_,__optional_) – The layout enum of the input tensor C. Defaults to None.
- **elem_ty_c** (_Union__[__Type__[__Numeric__]__,__None__]__,__optional_) – The element type for input tensor C. Defaults to None.

**Returns:**
: Returns epilog tiler, which is used in subsequent epilog partitions.

**Return type:**
: cute.Tile

**Raises:**
: **ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

```
`cutlass.utils.sm100.``get_smem_store_op`(
```

Selects the largest vectorized smem store atom available subject to
constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

**Parameters:**
: - **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
- **elem_ty_d** (_Type__[__Numeric__]_) – The element type for output tensor D.
- **elem_ty_acc** (_Type__[__Numeric__]_) – The element type for accumulator.
- **tiled_tmem_load** ([_cute.TiledCopy_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – An instance of TiledCopy that represents the tmem load operation.

**Returns:**
: Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

**Return type:**
: [cute.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

```
`cutlass.utils.sm100.``get_tmem_load_op`(
```

Finds a performant TMEM_LOAD copy op for the selected epilogue
tile (epi_tile), element types, and tcgen05.mma instruction used.

**Parameters:**
: - **cta_tile_shape** (_cute.Shape_) – A tuple or list representing the dimensions of the CTA tile.
- **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
- **elem_ty_d** (_Type__[__Numeric__]_) – The element type for output tensor D.
- **elem_ty_acc** (_Type__[__Numeric__]_) – The element type for accumulation.
- **epi_tile** (_cute.Tile_) – The epilogue tile configuration.
- **use_2cta_instrs** (_bool_) – A flag indicating whether the configuration is for 2 SMs.

**Returns:**
: An instance of Sm100TmemLoad with the computed configuration.

**Return type:**
: [cute.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

**Raises:**
: **ValueError** – If the function cannot handle the given combination of accumulation
and dimension types, or if it cannot determine the appropriate configuration based on
the input parameters.

```
`cutlass.utils.sm100.``make_smem_layout_a`(
```

This function helps with:

1. Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **tiled_mma** ([_cute.TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – The tiled MMA used to partition tensor A
- **mma_tiler_mnk** (_cute.cute.Tile_) – The MMA tile shape
- **a_dtype** (_Type__[__Numeric__]_) – The element type for tensor A
- **num_stages** (_int_) – The number of pipeline stages for tensor A

**Returns:**
: SMEM layout for tensor A

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm100.``make_smem_layout_b`(
```

This function helps:

1. Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **tiled_mma** ([_cute.TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – The tiled MMA which is used to partition the B tensor.
- **mma_tiler_mnk** (_cute.cute.Tile_) – The MMA tile shape.
- **b_dtype** (_Type__[__Numeric__]_) – The element type for the B tensor.
- **num_stages** (_int_) – The stage of the B tensor.

**Returns:**
: SMEM layout for the B tensor.

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm100.``make_smem_layout_epi`(
```

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **epi_dtype** (_Type__[__Numeric__]_) – The element type for the epilog tensor.
- **epi_layout** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for the epilog tensor.
- **epi_tile** (_cute.cute.Tile_) – The epilogue tile shape.
- **epi_stage** (_int_) – The stage of the epilog tensor.

**Returns:**
: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm100.``make_trivial_tiled_mma`(
```

Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

**Parameters:**
: - **ab_dtype** (_type__[__Numeric__]_) – Data type of operands A and B.
- **a_leading_mode** ([_tcgen05.OperandMajorMode_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) – Leading dimension of operand A (1 for K, 0 for M/N).
- **b_leading_mode** ([_tcgen05.OperandMajorMode_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) – Leading dimension of operand B (1 for K, 0 for M/N).
- **acc_dtype** (_type__[__Numeric__]_) – Data type of the accumulator.
- **cta_group** ([_tcgen05.CtaGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) – The CTA group to use.
- **mma_tiler_mn** (_Tuple__[__int__,__int__]_) – The shape (M, N, K) of the MMA tiler.
- **a_source** ([_cutlass.cute.nvgpu.tcgen05.OperandSource_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) – The source of operand A (SMEM by default or TMEM).

**Returns:**
: A tiled MMA atom.

**Return type:**
: [cute.TiledMma](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

**Raises:**
: **TypeError** – If the data type is not supported.

```
`cutlass.utils.sm100.``make_blockscaled_trivial_tiled_mma`(
```

Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

**Parameters:**
: - **ab_dtype** (_type__[__Numeric__]_) – Data type of operands A and B.
- **a_leading_mode** ([_tcgen05.OperandMajorMode_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) – Leading dimension of operand A (1 for K, 0 for M/N).
- **b_leading_mode** ([_tcgen05.OperandMajorMode_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) – Leading dimension of operand B (1 for K, 0 for M/N).
- **sf_dtype** (_type__[__Numeric__]_) – Data type of the Scale Factor.
- **sf_vec_size** (_int_) – The vector size of the Scale Factor.
- **cta_group** ([_tcgen05.CtaGroup_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) – The CTA group to use.
- **mma_tiler_mn** (_Tuple__[__int__,__int__]_) – The shape (M, N, K) of the MMA tiler.
- **a_source** ([_cutlass.cute.nvgpu.tcgen05.OperandSource_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) – The source of operand A (SMEM by default or TMEM).

**Returns:**
: A tiled MMA atom.

**Return type:**
: [cute.TiledMma](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

**Raises:**
: **TypeError** – If the data type is not supported.

```
`cutlass.utils.sm100.``cluster_shape_to_tma_atom_A`(
```

Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

**Parameters:**
: - **cluster_shape_mnk** (_cute.Shape_) – The shape of the cluster
- **atom_thr_id** (_cute.Layout_) – The thread ID of the atom

**Returns:**
: The appropriate TMA copy atom kind

**Return type:**
: [cpasync.CopyBulkTensorTileG2SMulticastOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

**Raises:**
: - **ValueError** – If the atom_sm_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

```
`cutlass.utils.sm100.``cluster_shape_to_tma_atom_B`(
```

Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

**Parameters:**
: - **cluster_shape_mnk** (_cute.Shape_) – The shape of the cluster
- **atom_thr_id** (_cute.Layout_) – The thread ID of the atom

**Returns:**
: The appropriate TMA copy atom kind

**Return type:**
: [cpasync.CopyBulkTensorTileG2SMulticastOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

**Raises:**
: - **ValueError** – If the atom_sm_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

```
`cutlass.utils.sm100.``cluster_shape_to_tma_atom_SFB`(
```

Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

**Parameters:**
: - **cluster_shape_mnk** (_cute.Shape_) – The shape of the cluster
- **atom_thr_id** (_cute.Layout_) – The thread ID of the atom

**Returns:**
: The appropriate TMA copy atom kind

**Return type:**
: [cpasync.CopyBulkTensorTileG2SMulticastOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

**Raises:**
: - **ValueError** – If the atom_sm_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

```
`cutlass.utils.sm100.``get_permutation_mnk`(
```

Get the permutation of M, N, K for the tiled MMA.

**Parameters:**
: - **tile_shape_mnk** (_cute.Shape_) – The shape of the tile
- **sf_vec_size** (_int_) – The vector size of the Scale Factor.
- **use_mxf8f6f4** (_bool_) – Whether to use MXF8F6F4 or MXF4NVF4.

**Returns:**
: The permutation of M, N, K

**Return type:**
: Tuple[int, int, int]

**Raises:**
: **ValueError** – If the tile shape is not divisible by the sf_vec_size

```
`cutlass.utils.sm100.``get_num_tmem_alloc_cols`(
```
