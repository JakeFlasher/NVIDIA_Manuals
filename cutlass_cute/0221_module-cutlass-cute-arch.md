---
title: "API documentation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html#module-cutlass.cute.arch"
---

## [API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.arch)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.arch "Permalink to this headline")

```
`cutlass.cute.arch.``make_warp_uniform`(
```

Provides a compiler hint indicating that the specified value is invariant across all threads in the warp,
which may enable performance optimizations.

**Parameters:**
: **value** (_Int_) – The integer value to be marked as warp-uniform.

**Returns:**
: The input value, marked as warp-uniform.

**Return type:**
: Int32

```
`cutlass.cute.arch.``elect_one`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `IfOpRegion`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.elect_one "Link to this definition")
```

Elects one thread within a warp.

```python
with elect_one():
    # Only one thread in the warp executes the code in this context
    pass
```

```
`cutlass.cute.arch.``mbarrier_init`(
```

Initializes a mbarrier with the specified thread arrival count.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **cnt** (_Int_) – The arrival count of the mbarrier

```
`cutlass.cute.arch.``mbarrier_init_fence`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.mbarrier_init_fence "Link to this definition")
```

A fence operation that applies to the mbarrier initializations.

```
`cutlass.cute.arch.``mbarrier_arrive_and_expect_tx`(
```

Arrives on a mbarrier and expects a specified number of transaction bytes.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **bytes** (_Int_) – The number of transaction bytes
- **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
the mbarrier is converted to a remote address in the peer CTA’s
SMEM.

```
`cutlass.cute.arch.``mbarrier_expect_tx`(
```

Expects a specified number of transaction bytes without an arrive.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **bytes** (_Int_) – The number of transaction bytes
- **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
the mbarrier is converted to a remote address in the peer CTA’s
SMEM.

```
`cutlass.cute.arch.``mbarrier_wait`(
```

Waits on a mbarrier with a specified phase.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **phase** (_Int_) – The phase to wait for (either 0 or 1)

```
`cutlass.cute.arch.``mbarrier_try_wait`(
```

Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **phase** (_Int_) – The phase to wait for (either 0 or 1)

**Returns:**
: A boolean value indicating whether the wait operation was successful

**Return type:**
: Boolean

```
`cutlass.cute.arch.``mbarrier_conditional_try_wait`(
```

Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

**Parameters:**
: - **cond** – A boolean predicate
- **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **phase** (_Int_) – The phase to wait for (either 0 or 1)

**Returns:**
: A boolean value indicating whether the wait operation was successful

**Return type:**
: Boolean

```
`cutlass.cute.arch.``mbarrier_arrive`(
```

Arrives on an mbarrier.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
the mbarrier is converted to a remote address in the peer CTA’s
SMEM.

```
`cutlass.cute.arch.``lane_idx`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `cutlass.cute.typing.Int32`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.lane_idx "Link to this definition")
```

Returns the lane index of the current thread within the warp.

```
`cutlass.cute.arch.``warp_idx`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `cutlass.cute.typing.Int32`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.warp_idx "Link to this definition")
```

Returns the warp index within a CTA.

```
`cutlass.cute.arch.``thread_idx`(
```

Returns the thread index within a CTA.

```
`cutlass.cute.arch.``block_dim`(
```

Returns the number of threads in each dimension of the CTA.

```
`cutlass.cute.arch.``block_idx`(
```

Returns the CTA identifier within a grid.

```
`cutlass.cute.arch.``grid_dim`(
```

Returns the number of CTAs in each dimension of the grid.

```
`cutlass.cute.arch.``cluster_idx`(
```

Returns the cluster identifier within a grid.

```
`cutlass.cute.arch.``cluster_dim`(
```

Returns the number of clusters in each dimension of the grid.

```
`cutlass.cute.arch.``cluster_size`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `cutlass.cute.typing.Int32`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cluster_size "Link to this definition")
```

Returns the number of CTA within the cluster.

```
`cutlass.cute.arch.``block_in_cluster_idx`(
```

Returns the CTA index within a cluster across all dimensions.

```
`cutlass.cute.arch.``block_in_cluster_dim`(
```

Returns the dimensions of the cluster.

```
`cutlass.cute.arch.``block_idx_in_cluster`(
```

Returns the linearized identifier of the CTA within the cluster.

```
`cutlass.cute.arch.``barrier`(
```

Creates a barrier, optionally named.

```
`cutlass.cute.arch.``barrier_arrive`(
```

```
`cutlass.cute.arch.``sync_threads`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.sync_threads "Link to this definition")
```

Synchronizes all threads within a CTA.

```
`cutlass.cute.arch.``sync_warp`(
```

Performs a warp-wide sync with an optional mask.

```
`cutlass.cute.arch.``fence_acq_rel_cta`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.fence_acq_rel_cta "Link to this definition")
```

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

```
`cutlass.cute.arch.``fence_acq_rel_cluster`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.fence_acq_rel_cluster "Link to this definition")
```

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

```
`cutlass.cute.arch.``fence_acq_rel_gpu`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.fence_acq_rel_gpu "Link to this definition")
```

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

```
`cutlass.cute.arch.``fence_acq_rel_sys`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.fence_acq_rel_sys "Link to this definition")
```

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

```
`cutlass.cute.arch.``cp_async_commit_group`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cp_async_commit_group "Link to this definition")
```

Commits all prior initiated but uncommitted cp.async instructions.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group).

```
`cutlass.cute.arch.``cp_async_wait_group`(_`n`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cp_async_wait_group "Link to this definition")
```

Waits till only a specified numbers of cp.async groups are pending.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).

```
`cutlass.cute.arch.``cp_async_bulk_commit_group`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cp_async_bulk_commit_group "Link to this definition")
```

Commits all prior initiated but uncommitted cp.async.bulk instructions.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group).

```
`cutlass.cute.arch.``cp_async_bulk_wait_group`(
```

Waits till only a specified numbers of cp.async.bulk groups are pending.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

```
`cutlass.cute.arch.``cluster_wait`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cluster_wait "Link to this definition")
```

A cluster-wide wait operation.

```
`cutlass.cute.arch.``cluster_arrive`(_`*`_, _`aligned``=``None`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cluster_arrive "Link to this definition")
```

A cluster-wide arrive operation.

```
`cutlass.cute.arch.``cluster_arrive_relaxed`(_`*`_, _`aligned``=``None`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cluster_arrive_relaxed "Link to this definition")
```

A cluster-wide arrive operation with relaxed semantics.

```
`cutlass.cute.arch.``vote_ballot_sync`(
```

Performs a ballot operation across the warp.

It copies the predicate from each thread in mask into the corresponding bit position of
destination register d, where the bit position corresponds to the thread’s lane id.

**Parameters:**
: - **pred** (_Boolean_) – The predicate value for the current thread
- **mask** (_Int__,__optional_) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)

**Returns:**
: A 32-bit integer where each bit represents a thread’s predicate value

**Return type:**
: Int32

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

```
`cutlass.cute.arch.``vote_any_sync`(
```

True if source predicate is True for any non-exited threads in mask. Negate the source
predicate to compute .none.

**Parameters:**
: - **pred** (_Boolean_) – The predicate value for the current thread
- **mask** (_Int__,__optional_) – A 32-bit integer mask specifying which threads participate, defaults to all
threads (0xFFFFFFFF)

**Returns:**
: A boolean value indicating if the source predicate is True for all non-exited
threads in mask

**Return type:**
: Boolean

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

```
`cutlass.cute.arch.``vote_all_sync`(
```

True if source predicate is True for all non-exited threads in mask. Negate the source
predicate to compute .none.

**Parameters:**
: - **pred** (_Boolean_) – The predicate value for the current thread
- **mask** (_Int__,__optional_) – A 32-bit integer mask specifying which threads participate, defaults to all
threads (0xFFFFFFFF)

**Returns:**
: A boolean value indicating if the source predicate is True for all non-exited
threads in mask

**Return type:**
: Boolean

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

```
`cutlass.cute.arch.``vote_uni_sync`(
```

True f source predicate has the same value in all non-exited threads in mask. Negating
the source predicate also computes .uni

**Parameters:**
: - **pred** (_Boolean_) – The predicate value for the current thread
- **mask** (_Int__,__optional_) – A 32-bit integer mask specifying which threads participate, defaults to all
threads (0xFFFFFFFF)

**Returns:**
: A boolean value indicating if the source predicate is True for all non-exited
threads in mask

**Return type:**
: Boolean

```
`cutlass.cute.arch.``warp_redux_sync`(
```

Perform warp-level reduction operation across threads.

Reduces values from participating threads in a warp according to the specified operation.
All threads in the mask receive the same result.

**Parameters:**
: - **value** (_Numeric_) – Input value to reduce
- **kind** (_Literal__[__"add"__,__"and"__,__"max"__,__"min"__,__"or"__,__"xor"__,__"fmin"__,__"fmax"__]_) – Reduction operation. Supported operations:
- Integer types (Int32/Uint32): “add”, “and”, “max”, “min”, “or”, “xor”
- Float types (Float32): “fmax”, “fmin” (or “max”/”min” which auto-convert to “fmax”/”fmin”)
- **mask_and_clamp** (_Int_) – Warp participation mask (default: FULL_MASK = 0xFFFFFFFF)
- **abs** (_bool_) – Apply absolute value before reduction (float types only)
- **nan** (_Optional__[__bool__]_) – Enable NaN propagation for fmax/fmin operations (float types only)

**Returns:**
: Reduced value (same for all participating threads)

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_max_float32`(
```

Performs an atomic max operation on a float32 value in global memory.

This implementation works correctly for non-negative values (>= 0) using direct bitcast.

**Parameters:**
: - **ptr** – Pointer to the memory location
- **value** (_Float32_) – The float32 value to compare and potentially store (should be >= 0 for correct results)
- **positive_only** (_bool_) – If True (default), assumes input values are non-negative.
This parameter is provided for API compatibility and future extensions.

**Returns:**
: The old value at the memory location

**Return type:**
: Float32

```
`cutlass.cute.arch.``atomic_add`(
```

Performs an atomic addition operation.

Atomically adds *val* to the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Union__[__Numeric__,__ir.Value__]_) – Value to add (scalar Numeric or vector ir.Value)
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Union[Numeric, ir.Value]

```
`cutlass.cute.arch.``atomic_and`(
```

Performs an atomic bitwise AND operation.

Atomically computes bitwise AND of *val* with the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value for AND operation
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_or`(
```

Performs an atomic bitwise OR operation.

Atomically computes bitwise OR of *val* with the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value for OR operation
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_xor`(
```

Performs an atomic bitwise XOR operation.

Atomically computes bitwise XOR of *val* with the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value for XOR operation
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_max`(
```

Performs an atomic maximum operation.

Atomically computes maximum of *val* and the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value for MAX operation
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_min`(
```

Performs an atomic minimum operation.

Atomically computes minimum of *val* and the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value for MIN operation
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_exch`(
```

Performs an atomic exchange operation.

Atomically exchanges *val* with the value at memory location *ptr* and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location
- **val** (_Numeric_) – Value to exchange
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``atomic_cas`(
```

Performs an atomic compare-and-swap (CAS) operation.

Atomically compares the value at the memory location with *cmp*. If they are equal,
stores *val* at the memory location and returns the old value.

**Parameters:**
: - **ptr** – Pointer to memory location. Supports:
- ir.Value (LLVM pointer)
- cute.ptr (_Pointer instance)
- **cmp** (_Numeric_) – Value to compare against current memory value
- **val** (_Numeric_) – Value to store if comparison succeeds
- **sem** (_Optional__[__Literal__[__"relaxed"__,__"release"__,__"acquire"__,__"acq_rel"__]__]_) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
- **scope** (_Optional__[__Literal__[__"gpu"__,__"cta"__,__"cluster"__,__"sys"__]__]_) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)

**Returns:**
: Old value at memory location

**Return type:**
: Numeric

```
`cutlass.cute.arch.``store`(
```

Store a value to a memory location.

**Parameters:**
: - **ptr** – Pointer to store to. Supports:
- ir.Value (LLVM pointer)
- cute.ptr (_Pointer instance)
- **val** (_Union__[__Numeric__,__ir.Value__]_) – Value to store (scalar Numeric or vector ir.Value)
- **level1_eviction_priority** – L1 cache eviction policy string literal:
“evict_normal” : .level1::eviction_priority = .L1::evict_normal
“evict_first” : .level1::eviction_priority = .L1::evict_first
“evict_last” : .level1::eviction_priority = .L1::evict_last
“evict_no_allocate” : .level1::eviction_priority = .L1::no_allocate
“evict_unchanged” : .level1::eviction_priority = .L1::evict_unchanged
- **cop** – Store cache modifier string literal:
- **ss** – Shared memory space string literal:
“cta” : .ss = .shared::cta
“cluster” : .ss = .shared::cluster
None : .ss = .global
- **sem** – Memory semantic string literal:
- **scope** – Memory scope string literal:

```
`cutlass.cute.arch.``load`(
```

Load a value from a memory location.

**Parameters:**
: - **ptr** – Pointer to load from. Supports:
- ir.Value (LLVM pointer)
- cute.ptr (_Pointer instance)
- **dtype** (_Union__[__type__[__Numeric__]__,__ir.VectorType__]_) – Data type to load. Can be:
- Scalar: Numeric type class (Int8, Uint8, Int32, Float32, etc.)
- Vector: ir.VectorType for vectorized load (e.g., ir.VectorType.get([4], Int64.mlir_type))
- **sem** – Memory semantic string literal:
- **scope** – Memory scope string literal:
- **level1_eviction_priority** – L1 cache eviction policy string literal:
“evict_normal” : .level1::eviction_priority = .L1::evict_normal
“evict_first” : .level1::eviction_priority = .L1::evict_first
“evict_last” : .level1::eviction_priority = .L1::evict_last
“evict_no_allocate” : .level1::eviction_priority = .L1::no_allocate
“evict_unchanged” : .level1::eviction_priority = .L1::evict_unchanged
- **cop** – Load cache modifier string literal:
- **ss** – Shared memory space string literal:
“cta” : .ss = .shared::cta
“cluster” : .ss = .shared::cluster
None : .ss = .global
- **level_prefetch_size** – L2 cache prefetch size hint string literal:
“size_64b” : .level::prefetch_size = .L2::64B
“size_128b” : .level::prefetch_size = .L2::128B
“size_256b” : .level::prefetch_size = .L2::256B

**Returns:**
: Loaded value (scalar Numeric or vector ir.Value)

**Return type:**
: Union[Numeric, ir.Value]

```
`cutlass.cute.arch.``popc`(
```

Performs a population count operation.

```
`cutlass.cute.arch.``fence_proxy`(
```

Fence operation to ensure memory consistency between proxies.

**Parameters:**
: - **kind** (_Literal__[__"alias"__,__"async"__,__"async.global"__,__"async.shared"__,__"tensormap"__,__"generic"__]_) – Proxy kind string literal:
- “alias” : Alias proxy
- “async” : Async proxy
- “async.global” : Async global proxy
- “async.shared” : Async shared proxy
- “tensormap” : Tensormap proxy
- “generic” : Generic proxy
- **space** (_Optional__[__Literal__[__"cta"__,__"cluster"__]__]_) – Shared memory space scope string literal (optional):
- “cta” : CTA (Cooperative Thread Array) scope
- “cluster” : Cluster scope

```
`cutlass.cute.arch.``warpgroup_reg_alloc`(_`reg_count``:` `int`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.warpgroup_reg_alloc "Link to this definition")
```

```
`cutlass.cute.arch.``warpgroup_reg_dealloc`(_`reg_count``:` `int`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.warpgroup_reg_dealloc "Link to this definition")
```

```
`cutlass.cute.arch.``setmaxregister_increase`(_`reg_count``:` `int`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.setmaxregister_increase "Link to this definition")
```

```
`cutlass.cute.arch.``setmaxregister_decrease`(_`reg_count``:` `int`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.setmaxregister_decrease "Link to this definition")
```

```
`cutlass.cute.arch.``fmax`(
```

```
`cutlass.cute.arch.``rcp_approx`(
```

```
`cutlass.cute.arch.``exp2`(
```

```
`cutlass.cute.arch.``cvt_i8x4_to_f32x4`(_`src_vec4`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8x4_to_f32x4 "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_i8x2_to_f32x2`(_`src_vec2`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8x2_to_f32x2 "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_i8_bf16`(_`src_i8`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8_bf16 "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_i8x2_to_bf16x2`(_`src_vec2`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8x2_to_bf16x2 "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_i8x4_to_bf16x4`(_`src_vec4`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8x4_to_bf16x4 "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_f32x2_bf16x2`(_`src_vec2`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_f32x2_bf16x2 "Link to this definition")
```

```
`cutlass.cute.arch.``alloc_smem`(
```

Statically allocates SMEM.

**Parameters:**
: - **element_type** (_Type__[__Numeric__]_) – The pointee type of the pointer.
- **size_in_elems** (_int_) – The size of the allocation in terms of number of elements of the
pointee type
- **alignment** (_int_) – An optional pointer alignment for the allocation

**Returns:**
: A pointer to the start of the allocation

**Return type:**
: Pointer

```
`cutlass.cute.arch.``get_dyn_smem`(
```

Retrieves a pointer to a dynamic SMEM allocation.

**Parameters:**
: - **element_type** (_Type__[__Numeric__]_) – The pointee type of the pointer.
- **alignment** (_int_) – An optional pointer alignment, the result pointer is offset appropriately

**Returns:**
: A pointer to the start of the dynamic SMEM allocation with a correct
alignement

**Return type:**
: Pointer

```
`cutlass.cute.arch.``get_dyn_smem_size`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.get_dyn_smem_size "Link to this definition")
```

Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time.
This can be used for bounds checking during shared memory allocation.

**Returns:**
: The size of dynamic shared memory in bytes

**Return type:**
: int

```
`cutlass.cute.arch.``get_max_tmem_alloc_cols`(_`compute_capability``:` `str`_) → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.get_max_tmem_alloc_cols "Link to this definition")
```

Get the tensor memory capacity in columns for a given compute capability.

Returns the maximum TMEM capacity in columns available for the specified
GPU compute capability.

**Parameters:**
: **compute_capability** (_str_) – The compute capability string (e.g. “sm_100”, “sm_103”)

**Returns:**
: The TMEM capacity in columns

**Return type:**
: int

**Raises:**
: **ValueError** – If the compute capability is not supported

```
`cutlass.cute.arch.``get_min_tmem_alloc_cols`(_`compute_capability``:` `str`_) → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.get_min_tmem_alloc_cols "Link to this definition")
```

Get the minimum TMEM allocation columns for a given compute capability.

Returns the minimum TMEM allocation columns available for the specified
GPU compute capability.

**Parameters:**
: **compute_capability** (_str_) – The compute capability string (e.g. “sm_100”, “sm_103”)

**Returns:**
: The minimum TMEM allocation columns

**Return type:**
: int

**Raises:**
: **ValueError** – If the compute capability is not supported

```
`cutlass.cute.arch.``retrieve_tmem_ptr`(
```

Retrieves a pointer to TMEM with the provided element type and alignment.

**Parameters:**
: - **element_type** (_Type__[__Numeric__]_) – The pointee type of the pointer.
- **alignment** (_int_) – The alignment of the result pointer
- **ptr_to_buffer_holding_addr** (_Pointer_) – A pointer to a SMEM buffer holding the TMEM address of the
start of the allocation allocation

**Returns:**
: A pointer to TMEM

**Return type:**
: Pointer

```
`cutlass.cute.arch.``alloc_tmem`(
```

Allocates TMEM.

**Parameters:**
: - **num_columns** (_Int_) – The number of TMEM columns to allocate
- **smem_ptr_to_write_address** (_Pointer_) – A pointer to a SMEM buffer where the TMEM address is written
to
- **is_two_cta** – Optional boolean parameter for 2-CTA MMAs
- **arch** (_str_) – The architecture of the GPU.

```
`cutlass.cute.arch.``relinquish_tmem_alloc_permit`(
```

Relinquishes the right to allocate TMEM so that other CTAs potentially in a different grid can
allocate.

```
`cutlass.cute.arch.``dealloc_tmem`(
```

Deallocates TMEM using the provided pointer and number of columns.

**Parameters:**
: - **tmem_ptr** (_Pointer_) – A pointer to the TMEM allocation to de-allocate
- **num_columns** (_Int_) – The number of columns in the TMEM allocation
- **is_two_cta** – Optional boolean parameter for 2-CTA MMAs

```
`cutlass.cute.arch.``prmt`(_`src`_, _`src_reg_shifted`_, _`prmt_indices`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.prmt "Link to this definition")
```

```
`cutlass.cute.arch.``cvt_i8_bf16_intrinsic`(_`vec_i8`_, _`length`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.arch.cvt_i8_bf16_intrinsic "Link to this definition")
```

Fast conversion from int8 to bfloat16. It converts a vector of int8 to a vector of bfloat16.

**Parameters:**
: - **vec_i8** (_1D vector__of__int8_) – The input vector of int8.
- **length** (_int_) – The length of the input vector.

**Returns:**
: The output 1D vector of bfloat16 with the same length as the input vector.

**Return type:**
: 1D vector of bfloat16

```
`cutlass.cute.arch.``cvt_i4_bf16_intrinsic`(
```

Fast conversion from int4 to bfloat16. It converts a vector of int4 to a vector of bfloat16.

**Parameters:**
: - **vec_i4** (_1D vector__of__int4_) – The input vector of int4.
- **length** (_int_) – The length of the input vector.
- **with_shuffle** (_bool_) – Whether the input vec_i4 follows a specific shuffle pattern.
If True, for consecutive 8 int4 values with indices of (0, 1, 2, 3, 4, 5, 6, 7),
the input elements are shuffled to (0, 2, 1, 3, 4, 6, 5, 7). For tailing elements less than 8,
the shuffle pattern is (0, 2, 1, 3) for 4 elements. No shuffle is needed for less than 4 elements.
Shuffle could help to produce converted bf16 values in the natural order of (0, 1, 2 ,3 ,4 ,5 ,6 ,7)
without extra prmt instructions and thus better performance.

**Returns:**
: The output 1D vector of bfloat16 with the same length as the input vector.

**Return type:**
: 1D vector of bfloat16

```
`cutlass.cute.arch.``issue_clc_query`(
```

The clusterlaunchcontrol.try_cancel instruction requests atomically cancelling the launch
of a cluster that has not started running yet. It asynchronously writes an opaque response
to shared memory indicating whether the operation succeeded or failed. On success, the
opaque response contains the ctaid of the first CTA of the canceled cluster.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier address in SMEM
- **clc_response_ptr** (_Pointer_) – A pointer to the cluster launch control response address in SMEM

```
`cutlass.cute.arch.``clc_response`(
```

After loading response from clusterlaunchcontrol.try_cancel instruction into 16-byte
register, it can be further queried using clusterlaunchcontrol.query_cancel instruction.
If the cluster is canceled successfully, predicate p is set to true; otherwise, it is
set to false. If the request succeeded, clusterlaunchcontrol.query_cancel.get_first_ctaid
extracts the CTA id of the first CTA in the canceled cluster. By default, the instruction
returns a .v4 vector whose first three elements are the x, y and z coordinate of first CTA
in canceled cluster.

**Parameters:**
: **result_addr** (_Pointer_) – A pointer to the cluster launch control response address in SMEM
