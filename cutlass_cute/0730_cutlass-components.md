---
title: "CUTLASS Components"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#cutlass-components"
---

## [CUTLASS Components](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-components)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-components "Permalink to this headline")

CUTLASS defines the following CUDA C++ templates to implement Implicit GEMM Convolution which are described in greater detail in subsequent sections.

**Activations tile iterators** load the activations tile into registers. Two implementations are provided:

- [conv2d_fprop_activation_tile_access_iterator_analytic.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h) computes pointer deltas and masks analytically
- [conv2d_fprop_activation_tile_access_iterator_optimized.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h) optimizes iterating over global memory and
creating GEMM-A tile in shared memory.

**Filter tile iterators** load filters into registers. Similarly, two implementations are provided:

- [conv2d_fprop_filter_tile_access_iterator_analytic.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h) computes pointer deltas and masks analytically
- [conv2d_fprop_filter_tile_access_iterator_optimized.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h) optimizes iterating over global memory and
creating GEMM-B tile in shared memory.

The improvements covered by optimized iterators are:

a. Precomputing kernel-invariant pointer deltas on the host
b. Computing cta-invariant mask predicates on device-side iterator ctors
c. Use of [fast divmod](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/fast_math.h) to map GEMM dimensions to convolution tensors.

For example, an _optimized_ activation iterator uses fast divmod to map GEMM _M_ to NPQ.

**Pipelined mainloop** loads threadblock-scoped tiles from global memory into shared memory and then applies
CUTLASS warp-level GEMM operations to load from Shared Memory and issue instructions to Turing Tensor Cores.

- [mma_pipelined.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/implicit_gemm_pipelined.h)

Operations for storing to shared memory and performing warp-wide matrix multiply operations using
Turing Tensor Cores are applied directly from the CUTLASS GEMM components. These include the
following components.

**Regular Tile Iterator** implemented in
[transform::threadblock::RegularTileIterator](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/transform/threadblock/regular_tile_iterator.h)
stores register-backed fragments to Shared Memory in permuted layouts.

**Warp-level GEMM** defined in [cutlass::gemm::warp::MmaTensorOp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/warp/mma_tensor_op.h)
defines tile iterators to load from Shared Memory and issue math instructions to Turing Tensor Cores.
Further details are [described in here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#warp-level-matrix-multiply-api).

**Epilogue** reorders accumulator elements among threads within a threadblock to efficiently update
the output tensor. It is implemented in [epilogue::threadblock::Epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/epilogue.h).
