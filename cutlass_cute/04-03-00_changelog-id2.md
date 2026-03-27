---
title: "4.3.0 (2025-10-20)"
section: "4.3.0"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html#changelog--id2"
---

## [4.3.0 (2025-10-20)](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#id2)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#id2 "Permalink to this headline")

- Debuggability improvements:
  - Supported source location tracking for DSL APIs
  - Supported dumping PTX and CUBIN
- Removed deprecated `cutlass.<arch>_utils.SMEM_CAPACITY["<arch_str>"]` and `cutlass.utils.ampere_helpers`
- Supported calling nested functions without capturing variables inside dynamic control flow
- Replaced usage of `cute.arch.barrier` in examples with corresponding APIs in `pipeline`
  - Use `pipeline.sync` for simple cases like synchronizing the whole CTA
  - Use `pipeline.NamedBarrier` to customize barriers with different participating threads and barrier id
- Added new APIs `repeat` and `repeat_as_tuple`
- Added new APIs `make_rmem_tensor` to create tensor in register memory (replace `make_fragment` with better naming)
- Added new APIs `make_rmem_tensor_like` which create rmem tensor from a tensor using the same shape with compact col-major strides
- Added `TmemAllocator` for allocating tensor memory
- Updated `SmemAllocator.allocate` to support allocation of a single scalar value
- Fixed `TensorSSA.reduce` to support static value as initial value
- Updated docstring for following APIs to be more concise and easier to understand:
  - `make_layout_tv`
  - `is_static`
  - `PipelineAsync`
  - `SmemAllocator`
- Fixed documentation for `pipeline`, `utils` and `cute.math` (`cute.math` is part of top level documentation)
