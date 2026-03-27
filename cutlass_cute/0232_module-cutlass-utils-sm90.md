---
title: "Utilities for SM90"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm90.html#module-cutlass.utils.sm90"
---

# [Utilities for SM90](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.utils.sm90)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.utils.sm90 "Permalink to this headline")

```
`cutlass.utils.sm90.``get_smem_store_op`(
```

Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

## [Parameters:](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#parameters)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#parameters "Permalink to this headline")

**layout_dLayoutEnum**

  The layout enum of the output tensor D.

**elem_ty_dType[Numeric]**

  The element type for output tensor D.

**elem_ty_accType[Numeric]**

  The element type for accumulator.

## [Returns:](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#returns)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#returns "Permalink to this headline")

Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

```
`cutlass.utils.sm90.``make_smem_layout_a`(
```

This function helps with:

1. Get the partitioned shape of the A tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **a_layout** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for tensor A
- **mma_tiler_mnk** (_cute.cute.Tile_) – The MMA tile shape
- **a_dtype** (_Type__[__Numeric__]_) – The element type for tensor A
- **num_stages** (_int_) – The number of pipeline stages for tensor A

**Returns:**
: SMEM layout for tensor A

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm90.``make_smem_layout_b`(
```

This function helps with:

1. Get the partitioned shape of the B tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **b_layout** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for tensor B
- **mma_tiler_mnk** (_cute.cute.Tile_) – The MMA tile shape
- **b_dtype** (_Type__[__Numeric__]_) – The element type for tensor B
- **num_stages** (_int_) – The number of pipeline stages for tensor B

**Returns:**
: SMEM layout for tensor B

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm90.``make_smem_layout_epi`(
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
- **smem_trg_shape** (_cute.Layout__|__None_) – Target shape for SMEM layout (optional).
- **smem_order** (_tuple__|__None_) – Order for SMEM layout (optional).

**Returns:**
: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.sm90.``compute_tile_shape_or_override`(
```

Compute the epilogue tile shape or use override if provided.

**Parameters:**
: - **tile_shape_mnk** (_Tuple__[__int__,__int__,__int__]_) – CTA tile shape (M,N,K)
- **element_type** (_type__[__Numeric__]_) – Data type of elements
- **is_cooperative** (_bool_) – Whether to use cooperative approach
- **epi_tile_override** (_Tuple__[__int__,__int__] or__None_) – Optional override for epilogue tile shape

**Returns:**
: Computed epilogue tile shape

**Return type:**
: Tuple[int, int]
