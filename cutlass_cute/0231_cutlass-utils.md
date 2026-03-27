---
title: "cutlass.utils"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html#cutlass-utils"
---

# [cutlass.utils](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#cutlass-utils)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass-utils "Permalink to this headline")

The `cutlass.utils` module contains utilities for developing kernels with CuTe DSL.

```
`cutlass.utils.``get_smem_capacity_in_bytes`(
```

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

**Parameters:**
: **compute_capability** (_Optional__[__str__]_) – The compute capability string (e.g. “70”, “75”, “80”)

**Returns:**
: The shared memory capacity in bytes

**Return type:**
: int

**Raises:**
: **ValueError** – If the compute capability is not supported

```
_`class`_`cutlass.utils.``SmemAllocator`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.SmemAllocator "Link to this definition")
```

Bases: `object`

A helper class for managing shared memory allocation on GPU.

This class manages shared memory and provides APIs for allocation of raw bytes,
numeric types, arrays, and tensors with specified layouts and alignments.

> **Note**
>
> - The base pointer is aligned to 1024 bytes upon initialization.
> - There is no need to explicitly specify shared memory size in kernel launch.
> - Currently only supports static layouts. Dynamic layouts are not supported.

**Examples**:

```python
smem = SmemAllocator()

# Allocate raw bytes
buf_ptr = smem.allocate(100)  # 100 bytes

# Allocate numeric type
int8_ptr = smem.allocate(Int8)  # 1 byte

# Define a struct
@cute.struct
class SharedStorage:
    alpha: cutlass.Float32
    x: cutlass.Int32

# Allocate struct
struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

# use of struct members
struct_ptr.alpha = 1.0
struct_ptr.x = 2

# Allocate array
int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

# Allocate tensor
layout = cute.make_layout((16, 16))
tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes
```

```
_`static`_`capacity_in_bytes`(
```

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

**Parameters:**
: **compute_capability** (_Optional__[__str__]_) – The compute capability string (e.g. “70”, “75”, “80”)

**Returns:**
: The shared memory capacity in bytes

**Return type:**
: int

**Raises:**
: **ValueError** – If the compute capability is not supported

```
`__init__`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.SmemAllocator.__init__ "Link to this definition")
```

Initialize a new SmemAllocator instance.

Creates a new shared memory allocator with a base pointer aligned to 1024 bytes.
Tracks the allocator instance for memory management.

**Parameters:**
: - **loc** (_Optional__[__ir.Location__]_) – Source location information for debugging, defaults to None
- **ip** (_Optional__[__ir.InsertionPoint__]_) – Insertion point for MLIR operations, defaults to None

```
`allocate`(
```

```
`allocate`(
```

```
`allocate`(
```

Allocate a block of memory with specified size and alignment.

This method allocates a block of shared memory with the specified size and alignment requirements.
It supports allocating raw bytes, numeric types(as scalar value), and struct types.

**Parameters:**
: - **size_or_type** (_Union__[__int__,__Type__[__Numeric__]__,_[_cute.struct_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.struct "cutlass.cute.struct")_]_) – The allocation specification, which can be:
- An integer specifying the number of bytes to allocate
- A Numeric type (e.g., Int8, Float32) to allocate space for one element
- A struct type to allocate space for the entire struct
- **byte_alignment** (_int__,__optional_) – The minimum byte alignment requirement for the allocation, defaults to 1
- **loc** (_Optional__[__ir.Location__]_) – Source location information for debugging, defaults to None
- **ip** (_Optional__[__ir.InsertionPoint__]_) – Insertion point for MLIR operations, defaults to None

**Returns:**
: For raw bytes and numeric types, returns a pointer to the allocated memory.
For struct types, returns an initialized struct instance at the allocated location.

**Return type:**
: cute.Pointer

**Raises:**
: - **ValueError** – If size is negative or alignment is less than 1
- **TypeError** – If size_or_type is not an integer, Numeric type, or struct
- **RuntimeError** – If allocation would exceed available shared memory

```
`allocate_array`(
```

Allocate an array of elements in shared memory.

**Parameters:**
: - **element_type** (_Type__[__Numeric__]_) – The type of elements to allocate
- **num_elems** (_int__,__optional_) – Number of elements to allocate, defaults to 1

**Returns:**
: Pointer to the start of the allocated array

**Return type:**
: cute.Pointer

**Raises:**
: - **ValueError** – If num_elems is less than 1
- **TypeError** – If element_type is not a Numeric type

```
`allocate_tensor`(
```

Allocate a tensor in shared memory.

Note: Currently only supports static layouts. Dynamic layouts are not supported.

**Parameters:**
: - **element_type** (_Type__[__Numeric__]_) – The type of elements in the tensor
- **layout** (_Union__[__int__,__cute.Layout__,__cute.ComposedLayout__]_) – The layout specification for the tensor. Must be a static layout.
- **byte_alignment** (_int__,__optional_) – The byte alignment requirement, defaults to 1
- **swizzle** ([_cute.Swizzle_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.Swizzle "cutlass.cute.Swizzle")_,__optional_) – Swizzle for position-dependent swizzling, defaults to None

**Returns:**
: The allocated tensor with specified properties

**Return type:**
: cute.Tensor

**Raises:**
: - **TypeError** – If element_type is not a Numeric type or if swizzle conflicts with layout
- **ValueError** – If allocation is not byte-aligned
- **NotImplementedError** – If dynamic layout is specified

```
_`class`_`cutlass.utils.``TmemAllocator`(
```

Bases: `object`

A class for managing tensor memory allocation on GPUs.

This class manages allocation/deallocation of tensor memory, including the mbarrier
synchronization for two cta use case.

**Variables:**
: - **_alloc_result_dst_smem_ptr** – The smem pointer that holds the base address of allocated tensor memory.
- **_barrier_for_retrieve** – The barrier for retrieving tensor memory ptr.
- **_allocator_warp_id** – The warp id of the allocator warp.
- **_is_two_cta** – Whether the allocator is for two cta.
- **_num_allocated_columns** – The number of columns allocated in the tensor memory.
- **_two_cta_tmem_dealloc_mbar_ptr** – The mbarrier pointer required when deallocating tensor memory for two cta.
- **_arch** – The architecture of the GPU.

```
`__init__`(
```

Initialize a TmemAllocator instance for managing tensor memory on Blackwell GPUs.

This initializer sets up the allocator’s state, including the shared memory (smem) pointer
holding the base address of the allocated tensor memory, barrier synchronization for
retrieving the tensor memory pointer, allocator warp ID, whether the allocator is being used
for a 2-SM configuration, number of allocated columns in tensor
memory, and the optional mbarrier pointer for deallocation in the 2-SM case.

If *is_two_cta* is set to True, this will initialize the mbarrier pointer required for tensor
memory deallocation across two CTAs.

**Parameters:**
: - **alloc_result_dst_smem_ptr** (_cute.Pointer_) – The shared memory pointer that holds the base address of allocated tensor memory.
- **barrier_for_retrieve** ([_pipeline.NamedBarrier_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html#cutlass.pipeline.NamedBarrier "cutlass.pipeline.NamedBarrier")) – The named barrier for retrieving the tensor memory pointer.
- **allocator_warp_id** (_int__,__optional_) – The warp ID of the allocator warp, defaults to 0.
- **is_two_cta** (_bool__,__optional_) – Whether the allocator should coordinate two CTAs, defaults to False.
- **num_allocated_columns** (_int__,__optional_) – The number of columns allocated in tensor memory, defaults to 0.
- **two_cta_tmem_dealloc_mbar_ptr** (_cute.Pointer__,__optional_) – The mbarrier pointer required for two-CTA tensor memory deallocation, optional.
- **loc** (_Any__,__optional_) – Optional codegen location for debugging and error reporting.
- **ip** (_Any__,__optional_) – Optional insertion point for codegen.

**Raises:**
: **AssertionError** – If two_cta_tmem_dealloc_mbar_ptr is None while is_two_cta is True.

```
`check_valid_num_columns`(_`num_columns``:` `int`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TmemAllocator.check_valid_num_columns "Link to this definition")
```

Check if the number of columns is valid.

This method checks if the number of columns is valid.
It checks if the number of columns is larger than 0, smaller than max capacity, a multiple of 32, and a power of two.

```
`wait_for_alloc`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TmemAllocator.wait_for_alloc "Link to this definition")
```

Wait for the allocator warp to finish allocation.

This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.

```
`retrieve_ptr`(
```

Retrieve the pointer to the allocated tensor memory.

This method can be called by all warps after allocation has been performed
by the allocator warp.

```
`cutlass.utils.``get_num_tmem_alloc_cols`(
```

Get the total number of TMEM allocation columns for the given TMEM tensors.

**Parameters:**
: - **tmem_tensors** (_Union__[__cute.Tensor__,__List__[__cute.Tensor__]__]_) – The TMEM tensors to get the number of allocation columns for.
- **rounding** (_bool_) – Whether to round up the number of allocation columns to the nearest power of 2.
- **arch** (_str_) – The architecture of the GPU.

**Returns:**
: The total number of TMEM allocation columns.

**Return type:**
: int

**Raises:**
: **ValueError** – If the number of TMEM allocation columns exceeds the maximum capacity or is less than 32.

```
_`class`_`cutlass.utils.``LayoutEnum`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "Link to this definition")
```

Bases: `Enum`

An enumeration.

```
`ROW_MAJOR`_`=` `'row_major'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.ROW_MAJOR "Link to this definition")
```

```
`COL_MAJOR`_`=` `'col_major'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.COL_MAJOR "Link to this definition")
```

```
`mma_major_mode`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.mma_major_mode "Link to this definition")
```

```
`sm90_mma_major_mode`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.sm90_mma_major_mode "Link to this definition")
```

```
`is_k_major_a`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_k_major_a "Link to this definition")
```

```
`is_m_major_a`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_m_major_a "Link to this definition")
```

```
`is_n_major_b`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_n_major_b "Link to this definition")
```

```
`is_k_major_b`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_k_major_b "Link to this definition")
```

```
`is_n_major_c`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_n_major_c "Link to this definition")
```

```
`is_m_major_c`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum.is_m_major_c "Link to this definition")
```

```
_`static`_`from_tensor`(
```

```
_`class`_`cutlass.utils.``WorkTileInfo`(
```

Bases: `object`

A class to represent information about a work tile.

**Variables:**
: - **tile_idx** – The index of the tile.
- **is_valid_tile** – Whether the tile is valid.

```
`__init__`(
```

```
_`property`_`is_valid_tile`_`:` `cutlass.cutlass_dsl.Boolean`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.WorkTileInfo.is_valid_tile "Link to this definition")
```

Check latest tile returned by the scheduler is valid or not. Any scheduling
requests after all tasks completed will return an invalid tile.

**Returns:**
: The validity of the tile.

**Return type:**
: Boolean

```
_`property`_`tile_idx`_`:` `cutlass.cute.typing.Coord`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.WorkTileInfo.tile_idx "Link to this definition")
```

Get the index of the tile.

**Returns:**
: The index of the tile.

**Return type:**
: cute.Coord

```
_`class`_`cutlass.utils.``PersistentTileSchedulerParams`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "Link to this definition")
```

Bases: `object`

A class to represent parameters for a persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

**Variables:**
: - **cluster_shape_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
- **problem_layout_ncluster_mnl** – Layout of the problem in terms of
number of clusters in (m, n, l) dimensions.

```
`__init__`(
```

Initializes the PersistentTileSchedulerParams with the given parameters.

**Parameters:**
: - **problem_shape_ntile_mnl** (_cute.Shape_) – The shape of the problem in terms of
number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
- **cluster_shape_mnk** (_cute.Shape_) – The shape of the cluster in (m, n) dimensions.
- **swizzle_size** (_int_) – Swizzling size in the unit of cluster. 1 means no swizzle
- **raster_along_m** (_bool_) – Rasterization order of clusters. Only used when swizzle_size > 1.
True means along M, false means along N.

**Raises:**
: **ValueError** – If cluster_shape_k is not 1.

```
`get_grid_shape`(
```

Computes the grid shape based on the maximum active clusters allowed.

**Parameters:**
: **max_active_clusters** (_Int32_) – The maximum number of active clusters that
can run in one wave.

**Returns:**
: A tuple containing the grid shape in (m, n, persistent_clusters).
- m: self.cluster_shape_m.
- n: self.cluster_shape_n.
- persistent_clusters: Number of persistent clusters that can run.

```
_`class`_`cutlass.utils.``StaticPersistentTileScheduler`(
```

Bases: `object`

A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

**Variables:**
: - **params** – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
- **num_persistent_clusters** – Number of persistent clusters that can be launched
- **cta_id_in_cluster** – ID of the CTA within its cluster
- **_num_tiles_executed** – Counter for executed tiles
- **_current_work_linear_idx** – Current cluster index

```
`__init__`(
```

Initializes the StaticPersistentTileScheduler with the given parameters.

**Parameters:**
: - **params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
- **num_persistent_clusters** (_Int32_) – Number of persistent clusters that can be launched.
- **current_work_linear_idx** (_Int32_) – Current cluster index.
- **cta_id_in_cluster** (_cute.Coord_) – ID of the CTA within its cluster.
- **num_tiles_executed** (_Int32_) – Counter for executed tiles.

```
_`static`_`create`(
```

Initialize the static persistent tile scheduler.

**Parameters:**
: - **params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for the persistent
tile scheduler.
- **block_idx** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d block index in the format (bidx, bidy, bidz).
- **grid_dim** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d grid dimensions for kernel launch.

**Returns:**
: A StaticPersistentTileScheduler object.

**Return type:**
: [StaticPersistentTileScheduler](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.StaticPersistentTileScheduler")

```
_`static`_`get_grid_shape`(
```

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

**Parameters:**
: - **params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for grid shape calculation.
- **max_active_clusters** (_Int32_) – Maximum active clusters allowed.

**Returns:**
: The calculated 3d grid shape.

**Return type:**
: Tuple[Integer, Integer, Integer]

```
`_get_current_work_for_linear_idx`(
```

Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

**Parameters:**
: **current_work_linear_idx** (_Int32_) – The linear index of the current work.

**Returns:**
: An object containing information about the current tile coordinates
and validity status.

**Return type:**
: [WorkTileInfo](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

```
`_get_cluster_work_idx_with_fastdivmod`(
```

FastDivmod optimized CLUSTER coordinate calculation.

CRITICAL: This should mimic problem_layout_ncluster_mnl.get_hier_coord()
which returns CLUSTER coordinates, not tile coordinates!

**Parameters:**
: **current_work_linear_idx** (_Int32_) – Linear index in the work space

**Returns:**
: Cluster coordinates (m, n, l) or None if FastDivmod not available

**Return type:**
: Tuple[Int32, Int32, Int32] or None

```
`get_current_work`(
```

```
`initial_work_tile_info`(
```

```
`advance_to_next_work`(
```

```
_`property`_`num_tiles_executed`_`:` `cutlass.cutlass_dsl.Int32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.StaticPersistentTileScheduler.num_tiles_executed "Link to this definition")
```

```
_`class`_`cutlass.utils.``StaticPersistentRuntimeTileScheduler`(
```

Bases: [`StaticPersistentTileScheduler`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler")

A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

**Variables:**
: - **params** – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
- **num_persistent_clusters** – Number of persistent clusters that can be launched
- **cta_id_in_cluster** – ID of the CTA within its cluster
- **_num_tiles_executed** – Counter for executed tiles
- **_current_work_linear_idx** – Current cluster index

```
`__init__`(
```

Initializes the StaticPersistentTileScheduler with the given parameters.

**Parameters:**
: - **params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
- **num_persistent_clusters** (_Int32_) – Number of persistent clusters that can be launched.
- **current_work_linear_idx** (_Int32_) – Current cluster index.
- **cta_id_in_cluster** (_cute.Coord_) – ID of the CTA within its cluster.
- **num_tiles_executed** (_Int32_) – Counter for executed tiles.
- **inner_mode** (_int_) – The inner mode along which the linear index will be decomposed first.

```
_`static`_`create`(
```

Initialize the static persistent tile scheduler.

**Parameters:**
: - **params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for the persistent
tile scheduler.
- **block_idx** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d block index in the format (bidx, bidy, bidz).
- **grid_dim** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d grid dimensions for kernel launch.
- **inner_mode** (_int_) – The inner mode along which the linear index will be decomposed first.

**Returns:**
: A StaticPersistentRuntimeTileScheduler object.

**Return type:**
: [StaticPersistentRuntimeTileScheduler](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.StaticPersistentRuntimeTileScheduler "cutlass.utils.StaticPersistentRuntimeTileScheduler")

```
`_get_current_work_for_linear_idx`(
```

Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

**Parameters:**
: **current_work_linear_idx** (_Int32_) – The linear index of the current work.

**Returns:**
: An object containing information about the current tile coordinates
and validity status.

**Return type:**
: [WorkTileInfo](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

```
_`class`_`cutlass.utils.``TensorMapUpdateMode`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapUpdateMode "Link to this definition")
```

Bases: `Enum`

Enum class defining tensor map update modes.

Modes:
GMEM: Update tensormap in global memory
SMEM: Load tensormap from global memory to shared memory,
update it in shared memory, then store back to global memory

```
`GMEM`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapUpdateMode.GMEM "Link to this definition")
```

```
`SMEM`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapUpdateMode.SMEM "Link to this definition")
```

```
_`class`_`cutlass.utils.``TensorMapManager`(
```

Bases: `object`

Manages TensorMap operations including initialization and updates.
Provides utilities to convert tensormap pointer to across different memory spaces.

```
`tensormap_update_mode`_`:` [`TensorMapUpdateMode`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapManager.tensormap_update_mode "Link to this definition")
```

```
`bytes_per_tensormap`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TensorMapManager.bytes_per_tensormap "Link to this definition")
```

```
`get_tensormap_ptr`(
```

```
`fence_tensormap_initialization`(
```

```
`fence_tensormap_update`(
```

```
`__init__`(
```

```
_`class`_`cutlass.utils.``GroupSearchResult`(
```

Bases: `object`

The result of the group search for grouped gemm.

**Parameters:**
: - **group_idx** (_Int32_) – The result group index
- **cta_tile_idx_m** (_Int32_) – CTA tile index along M dimension after rasterization
- **cta_tile_idx_n** (_Int32_) – CTA tile index along N dimension after rasterization
- **problem_shape_m** (_Int32_) – The M dimension of the gemm problem
- **problem_shape_n** (_Int32_) – The N dimension of the gemm problem
- **problem_shape_k** (_Int32_) – The K dimension of the gemm problem
- **cta_tile_count_k** (_Int32_) – Number of tiles along K dimension

```
`__init__`(
```

```
_`class`_`cutlass.utils.``GroupedGemmGroupSearchState`(
```

Bases: `object`

The state of group index search for grouped gemm.

The state will be initialized once and updated in every round of group index search.

**Parameters:**
: - **start_group_idx** (_Int32_) – The group idx to start the search with
- **tile_count_prev_group** (_Int32_) – Number of tiles before the matched group
- **tile_count_searched** (_Int32_) – Number of tiles we have searched. When the matched group is found,
it records the number of tiles including the matched group

```
`__init__`(
```

```
`cutlass.utils.``create_initial_search_state`() → [`GroupedGemmGroupSearchState`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.create_initial_search_state "Link to this definition")
```

Create an initial search state for grouped gemm.

**Returns:**
: A new search state with initial values

**Return type:**
: [GroupedGemmGroupSearchState](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

```
_`class`_`cutlass.utils.``GroupedGemmTileSchedulerHelper`(_`**``kwargs`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupedGemmTileSchedulerHelper "Link to this definition")
```

Bases: `object`

A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA tile index for grouped gemm.

**Parameters:**
: - **group_count** (_int_) – Number of groups in current grouped gemm problem
- **tile_sched_params** ([_PersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameter used to create the tile scheduler this helper works with
- **cluster_tile_shape_mnk** (_tuple__[__int__,__int__,__int__]_) – The shape of cluster tile as (m, n, k)
- **search_state** ([_GroupedGemmGroupSearchState_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")) – The initial search state

```
`__init__`(
```

```
`delinearize_z`(
```

Delinearize the linear z index and return GroupSearchResult.

This function should be used by warps that need to know the CTA tile index on M and N dimensions.

**Parameters:**
: - **cta_tile_coord** (_tuple__of__Int32_) – The raw CTA coordinate from tile scheduler
- **problem_shape_mnkl** (_cute.Tensor_) – Tensor containing gemm problem size (M, N, K, L) for each group

**Returns:**
: The search result containing group index and tile coordinates

**Return type:**
: [GroupSearchResult](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupSearchResult "cutlass.utils.GroupSearchResult")

```
`search_cluster_tile_count_k`(
```

Search the matched group for given linear index and compute the number of tiles along K dimension for the matched group.

This function should be used by warps that are only interested in the number of tiles along K dimension.

**Parameters:**
: - **cta_tile_coord** (_tuple__of__Int32_) – The raw CTA coordinate from tile scheduler
- **problem_shape_mnkl** (_cute.Tensor_) – Tensor containing gemm problem size (M, N, K, L) for all groups

**Returns:**
: A tuple containing cluster count along K dimension and the group index

**Return type:**
: Tuple[Int32, Int32]

```
`_prefix_sum`(
```

Perform prefix sum within a full warp.

**Parameters:**
: **value_per_thread** (_Int32_) – The value for this thread to contribute to the prefix sum

**Returns:**
: The prefix sum result for this thread

**Return type:**
: Int32

```
`_get_problem_for_group`(
```

Load gemm problem (m,n,k,l) for the specified group from global memory to register.

**Parameters:**
: - **problem_shape_mnkl** (_cute.Tensor_) – Tensor in global memory with layout (group_count, 4):(4, 1)
- **group_idx** (_Int32_) – The index of the group to load

**Returns:**
: The problem shape tensor for the specified group

**Return type:**
: cute.Tensor

```
`_get_cluster_tile_count_mn`(
```

Compute total cluster count.

**Parameters:**
: **problem_shape** (_cute.Tensor_) – Tensor containing problem shape (m, n, k, l)

**Returns:**
: The total cluster tile count for M and N dimensions

**Return type:**
: Int32

```
`_compute_cta_tile_coord`(
```

Compute CTA tile indices along M and N dimensions based on the linear index within a group.

It uses the AlongM mode to decompose the linear index onto M and N dimensions.

**Parameters:**
: - **cluster_tile_idx** (_Int32_) – The linear index within a group
- **cta_tile_coord_in_cluster** (_tuple__of__Int32_) – CTA indices along M and N dimensions within a cluster
- **cluster_tile_count_m** (_Int32_) – The number of clusters along M dimension of the matched group
- **cluster_tile_count_n** (_Int32_) – The number of clusters along N dimension of the matched group

**Returns:**
: A tuple containing CTA tile indices along M and N dimensions

**Return type:**
: tuple of (Int32, Int32)

```
`_group_search`(
```

Search which group the linear index belongs to.

**Parameters:**
: - **linear_idx** (_Int32_) – The linear index to be decomposed
- **problem_shape_mnkl** (_cute.Tensor_) – Tensor containing gemm problem size (M, N, K, L) for all groups
- **init_group_idx** (_Int32_) – The group idx to start the search with
- **init_tile_count_searched** (_Int32_) – The number of tiles we have searched

**Returns:**
: The updated search state

**Return type:**
: [GroupedGemmGroupSearchState](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

```
`_group_search_and_load_problem_shape`(
```

Perform group search and load problem shape for the matched group.

**Parameters:**
: - **linear_idx** (_Int32_) – The linear index to be decomposed
- **problem_shape_mnkl** (_cute.Tensor_) – Tensor containing gemm problem size (M, N, K, L) for all groups
- **start_group_idx** (_Int32_) – The group idx to start the search with
- **tile_count_searched** (_Int32_) – The number of tiles we have searched

**Returns:**
: A tuple containing the final group index and the problem shape tensor

**Return type:**
: Tuple[Int32, cute.Tensor]

```
_`class`_`cutlass.utils.``HardwareInfo`(_`device_id``:` `int` `=` `0`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo "Link to this definition")
```

Bases: `object`

device_id: CUDA device ID to get the hardware info.

```
`__init__`(_`device_id``:` `int` `=` `0`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo.__init__ "Link to this definition")
```

```
`get_max_active_clusters`(
```

Get the maximum number of active clusters for a given cluster size.

When a stream from a green context is provided, the occupancy calculation
will reflect the reduced SM partition of the green context.

**Parameters:**
: - **cluster_size** (_int_) – Number of blocks per cluster (must be between 1 and 32)
- **stream** (_driver.CUstream__,__optional_) – Optional CUDA stream handle. If provided (especially from a green context),
the occupancy calculation reflects the stream’s SM partition.

**Returns:**
: Maximum number of active clusters

**Return type:**
: int

```
`get_l2_cache_size_in_bytes`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo.get_l2_cache_size_in_bytes "Link to this definition")
```

```
`get_device_multiprocessor_count`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo.get_device_multiprocessor_count "Link to this definition")
```

```
`_checkCudaErrors`(_`result`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._checkCudaErrors "Link to this definition")
```

```
`_cudaGetErrorEnum`(_`error`_) → `str`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._cudaGetErrorEnum "Link to this definition")
```

```
`_cuda_driver_version_ge`(_`major``:` `int`_, _`minor``:` `int`_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._cuda_driver_version_ge "Link to this definition")
```

```
`_cuda_driver_version_lt`(_`major``:` `int`_, _`minor``:` `int`_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._cuda_driver_version_lt "Link to this definition")
```

```
`_empty_kernel`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._empty_kernel "Link to this definition")
```

```
`_host_function`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._host_function "Link to this definition")
```

```
`_get_device_function`() → `cuda.bindings.driver.CUfunction`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.HardwareInfo._get_device_function "Link to this definition")
```

Get a device function by compiling a dummy kernel using cuteDSL pipeline.

```
_`class`_`cutlass.utils.``TransformMode`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TransformMode "Link to this definition")
```

Bases: `Enum`

An enumeration for the possible transform modes of a mixed-input GEMM.

```
`ConvertOnly`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TransformMode.ConvertOnly "Link to this definition")
```

```
`ConvertScale`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.TransformMode.ConvertScale "Link to this definition")
```

```
`cutlass.utils.``scale_tma_partition`(
```

Perform TMA partition for scale tensor.
This method partitions the global memory and shared memory buffer for the scale tensor for TMA load.
:param tCsS: Input scale shared memory tensor
:type tCsS: cute.Tensor
:param tCgS: Input scale global memory tensor
:type tCgS: cute.Tensor
:param tma_atom_s: TMA copy atom for scale tensor
:type tma_atom_s: cute.CopyAtom
:param block_in_cluster_coord_vmnk: CTA coord in the cluster
:type block_in_cluster_coord_vmnk: cute.Coord
:param scale_cta_layout: Layout of CTA from the view of the scale tensor
:type scale_cta_layout: cute.Layout
:return: A tuple containing (tSsS, tSgS) where:

> - tSsS: Partitioned scale tensor in shared memory
> - tSgS: Partitioned scale tensor in global memory

**Return type:**
: tuple[cute.Tensor, cute.Tensor]

```
`cutlass.utils.``transform_partition`(
```

Partition tensors for transform input and output.
This method sets up the copy atoms and partitions the shared/tensor memory
for the transformation of tensor A.
:param transform_a_source: Where the transformed tensor A is stored (TMEM or SMEM)
:type transform_a_source: tcgen05.OperandSource
:param scale_mode: The transform mode (ConvertOnly or ConvertScale)
:type scale_mode: TransformMode
:param copy_atom_a_input: Copy atom for loading A from shared memory
:type copy_atom_a_input: cute.CopyAtom
:param copy_atom_a_transform: Copy atom for storing transformed A
:type copy_atom_a_transform: cute.CopyAtom
:param sA_input: Input tensor A in shared memory
:type sA_input: cute.Tensor
:param A_transform: Transformed tensor A in tensor or shared memory
:type A_transform: cute.Tensor
:param transform_local_tidx: Local thread index for transformation warps
:type transform_local_tidx: cutlass.Int32
:return: A tuple containing (src_copy_a, dst_copy_a, tAsA_input, tA_transform) where:

> - src_copy_a: Tiled copy for source tensor
> - dst_copy_a: Tiled copy for destination tensor
> - tAsA_input: Partitioned input tensor A
> - tA_transform: Partitioned transformed tensor A

**Return type:**
: tuple[Optional[[cute.TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], Optional[[cute.TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], cute.Tensor, cute.Tensor]

```
`cutlass.utils.``scale_partition`(
```

Partition the scale tensor for transformation.
This method prepares the copy atom and partitions the shared memory for the scale tensor.
:param src_copy_a: Tiled copy for the source tensor
:type src_copy_a: cute.TiledCopy
:param tCsS: Scale tensor in shared memory
:type tCsS: cute.Tensor
:param transform_local_tidx: Local thread index for transformation warps
:type transform_local_tidx: cutlass.Int32
:param mma_dtype: Data type for the MMA operation
:type mma_dtype: type[cutlass.Numeric]
:return: A tuple containing (smem_thr_copy_S, tSsS_trans, tSrS_copy, tSrS) where:

> - smem_thr_copy_S: Tiled copy for the scale tensor
> - tSsS_trans: Partitioned scale tensor for transformation
> - tSrS_copy: Register fragment for the scale tensor
> - tSrS: View of scale tensor used for transformation computation

**Return type:**
: tuple[[cute.TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy"), cute.Tensor, cute.Tensor, cute.Tensor]

```
`cutlass.utils.``get_gmem_layout_scale`(
```

Get the layout of the scale tensor in global memory.
:param scale_shape_mkl: The shape of the scale tensor (M, K, L).
:type scale_shape_mkl: tuple[int, int, int]
:return: The layout of the scale tensor in global memory.
:rtype: cute.Layout

```
`cutlass.utils.``get_smem_layout_scale`(
```

Get the layout of the scale tensor in shared memory.
:return: A tuple containing (scale_tile_shape, smem_layout_scale_per_stage, smem_layout_scale) where:

> - scale_tile_shape: The tile shape
> - smem_layout_scale_per_stage: Shared memory layout for scale tensor per stage
> - smem_layout_scale: Shared memory layout for scale tensor

**Return type:**
: tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout]

```
`cutlass.utils.``compute_smem_layout`(
```

Compute shared memory layouts for tensor A, transformed A and tensor B.
:param tiled_mma: The tiled MMA object defining the core computation.
:type tiled_mma: cute.TiledMma
:param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
:type mma_tiler_mnk: tuple[int, int, int]
:param a_dtype: Data type of operand A.
:type a_dtype: type[cutlass.Numeric]
:param b_dtype: Data type of operand B.
:type b_dtype: type[cutlass.Numeric]
:param load2trans_stage_count: Number of stages for load-to-transform pipeline.
:type load2trans_stage_count: int
:param trans2mma_stage_count: Number of stages for transform-to-MMA pipeline.
:type trans2mma_stage_count: int
:return: A tuple containing (smem_layout_a, smem_layout_a_transform, smem_layout_b) where:

> - smem_layout_a: Shared memory layout for tensor A
> - smem_layout_a_transform: Shared memory layout for transformed tensor A
> - smem_layout_b: Shared memory layout for tensor B

**Return type:**
: tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]

```
`cutlass.utils.``get_transform_a_source`(
```

Determine the operand source for transformed A tensor based on the operand major mode.

```
`cutlass.utils.``get_tma_atom_kind`(
```

Get the TMA atom kind based on 1) whether it’s a multicast operation,
2) whether 2CTA tcgen05.mma instruction is enabled, and
3) whether it’s a B tensor

```
`cutlass.utils.``get_copy_atom_a_transform`(
```

Determine the copy atom for transformed A tensor based on the operand source and tile size.

```
`cutlass.utils.``is_valid_scale_granularity`(
```

Check if the scale granularity settings are valid for the given data type and problem size.

```
`cutlass.utils.``get_divisibility`(
```

Calculate the largest power of 2 divisibility factor for memory alignment.

```
`cutlass.utils.``compute_epilogue_tile_shape`(
```

Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

**Parameters:**
: - **cta_tile_shape** (_cute.Shape_) – A tuple or list representing the dimensions of the CTA tile, where
cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
corresponds to the width (N) of the tile.
- **use_2cta_instrs** (_bool_) – A flag indicating whether the configuration is for a 2SM setup.
- **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
- **elem_ty_d** (_Type__[__Numeric__]_) – The element type of output tensor D.
- **layout_c** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")_,__optional_) – The layout enum of the input tensor C. Defaults to None.
- **elem_ty_c** (_Union__[__Type__[__Numeric__]__,__None__]__,__optional_) – The element type for input tensor C. Defaults to None.

**Returns:**
: Returns epilog tiler, which is used in subsequent epilog partitions.

**Return type:**
: cute.Tile

**Raises:**
: **ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

```
`cutlass.utils.``get_smem_store_op`(
```

Selects the largest vectorized smem store atom available subject to
constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

**Parameters:**
: - **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
- **elem_ty_d** (_Type__[__Numeric__]_) – The element type for output tensor D.
- **elem_ty_acc** (_Type__[__Numeric__]_) – The element type for accumulator.
- **tiled_tmem_load** ([_cute.TiledCopy_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – An instance of TiledCopy that represents the tmem load operation.

**Returns:**
: Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

**Return type:**
: [cute.CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

```
`cutlass.utils.``get_tmem_load_op`(
```

Finds a performant TMEM_LOAD copy op for the selected epilogue
tile (epi_tile), element types, and tcgen05.mma instruction used.

**Parameters:**
: - **cta_tile_shape** (_cute.Shape_) – A tuple or list representing the dimensions of the CTA tile.
- **layout_d** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
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
`cutlass.utils.``make_smem_layout_a`(
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
`cutlass.utils.``make_smem_layout_b`(
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
`cutlass.utils.``make_smem_layout_epi`(
```

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

**Parameters:**
: - **epi_dtype** (_Type__[__Numeric__]_) – The element type for the epilog tensor.
- **epi_layout** ([_LayoutEnum_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for the epilog tensor.
- **epi_tile** (_cute.cute.Tile_) – The epilogue tile shape.
- **epi_stage** (_int_) – The stage of the epilog tensor.

**Returns:**
: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

**Return type:**
: Union[cute.Layout, cute.ComposedLayout]

```
`cutlass.utils.``make_trivial_tiled_mma`(
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
`cutlass.utils.``make_blockscaled_trivial_tiled_mma`(
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
_`class`_`cutlass.utils.``ClcDynamicPersistentTileSchedulerParams`(
```

Bases: `object`

A class to represent parameters for a dynamic persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

**Variables:**
: **cluster_shape_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).

```
`__init__`(
```

Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

**Parameters:**
: - **problem_shape_ntile_mnl** (_cute.Shape_) – The shape of the problem in terms of
number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
- **cluster_shape_mnk** (_cute.Shape_) – The shape of the cluster in (m, n) dimensions.

**Raises:**
: **ValueError** – If cluster_shape_k is not 1.

```
`get_grid_shape`(
```

Computes the grid shape based on the problem shape and cluster shape.

**Returns:**
: the grid is the CTA numbers that has aligned with cluster shape.

```
_`class`_`cutlass.utils.``ClcDynamicPersistentTileScheduler`(
```

Bases: `object`

A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

**Variables:**
: - **params** – Tile schedule related params, including cluster shape.
- **cta_id_in_cluster** – ID of the CTA within its cluster
- **_num_tiles_executed** – Counter for executed tiles

```
`__init__`(
```

Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

**Parameters:**
: - **params** ([_ClcDynamicPersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape.
- **cta_id_in_cluster** (_cute.Coord_) – ID of the CTA within its cluster.
- **num_tiles_executed** (_Int32_) – Counter for executed tiles.
- **clc_response_ptr** (_cute.Pointer_) – Pointer of the clc rsponse.
- **block_idx** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The block index.

```
`create`(
```

Initialize the dynamic persistent tile scheduler.

**Parameters:**
: - **params** ([_ClcDynamicPersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Parameters for the persistent
tile scheduler.
- **block_idx** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d block index in the format (bidx, bidy, bidz).
- **grid_dim** (_Tuple__[__Integer__,__Integer__,__Integer__]_) – The 3d grid dimensions for kernel launch.

**Returns:**
: A ClcDynamicPersistentTileScheduler object.

**Return type:**
: [ClcDynamicPersistentTileScheduler](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.ClcDynamicPersistentTileScheduler")

```
`get_grid_shape`(
```

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

**Parameters:**
: **params** ([_ClcDynamicPersistentTileSchedulerParams_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Parameters for grid shape calculation.

**Returns:**
: The calculated 3d grid shape.

**Return type:**
: Tuple[Integer, Integer, Integer]

```
`work_tile_info_from_clc_response`(
```

Simulates parsing CLC response data in Python.
result_addr: 16-byte response data (simulating shared memory access)

```
`get_current_work`(
```

```
`initial_work_tile_info`(
```

```
`advance_to_next_work`(
```

```
_`property`_`num_tiles_executed`_`:` `cutlass.cutlass_dsl.Int32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.ClcDynamicPersistentTileScheduler.num_tiles_executed "Link to this definition")
```

```
`cutlass.utils.``print_latex`(
```

Prints a layout.
:param x: A layout
:type x: Union[Layout, ComposedLayout]
:param color: A function that returns TiKZ colors
:type color: Callable

```
`cutlass.utils.``print_latex_tv`(
```

Prints a tv layout for a tile M N. Everything must be static.
:param layout_tv: A static thread value layout
:type layout_tv: Union[Layout, ComposedLayout]
:param tile_mn: A static M N tile
:type tile_mn: Union[IntTuple, Layout]
:param color: A function that returns TiKZ colors
:type color: Callable

```
`cutlass.utils.``is_fp8_dtype`(_`dtype``:` `Type``[``cutlass.cute.typing.Numeric``]`_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.utils.is_fp8_dtype "Link to this definition")
```

Check if dtype is a float8 type that doesn’t support dlpack.
params dtype: The cutlass numeric type to check
type dtype: Type[cutlass.Numeric]
return: True if the dtype is Float8E5M2 or Float8E4M3FN, False otherwise

```
`cutlass.utils.``create_cute_tensor_for_fp8`(
```

Create cute tensor, handling float8 types that don’t support dlpack.

For float8 types, the storage_tensor should be uint8 (for DLPack compatibility).
The source_f32_tensor provides the actual float32 values to convert to fp8.

params storage_tensor: Tensor for DLPack (uint8 for fp8, otherwise the actual dtype)
params dtype: Target cutlass dtype
params leading_dim: Leading dimension for dynamic layout
paramas source_f32_tensor: Float32 source data for fp8 conversion (required for fp8)
return: A cute tensor with the appropriate dtype and layout
