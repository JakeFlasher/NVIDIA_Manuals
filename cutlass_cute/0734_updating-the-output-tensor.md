---
title: "Updating the Output Tensor"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#updating-the-output-tensor"
---

### [Updating the Output Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#updating-the-output-tensor)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#updating-the-output-tensor "Permalink to this headline")

After the mainloop terminates, the accumulator tile of the warp-level GEMM stores a warp’s contribution to the output
tensor. However, the distribution of data among threads within the threadblock is specialized for efficient matrix multiply-accumulate
operations using Tensor Cores and is not conducive to efficient, coalesced operations to Global Memory. A data rearrangement is
needed.

The **Epilogue** is the component for exchanging accumulator elements through Shared Memory, loading slices of the output
matrix or tensor, applying an elementwise operation such as linear scaling or bias, and storing the result to the output tensor.
CUTLASS structures this as several components:

- [cutlass::epilogue::threadblock::Epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/epilogue.h) - the top-level component for looping over the entire threadblock tile
- [cutlass::epilogue::warp::TileIteratorTensorOp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h) - a specialized component for storing accumulators for Tensor Core to Shared Memory
- [cutlass::epilogue::threadblock::SharedLoadIterator](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/shared_load_iterator.h) - a component for loading elements from a row-major arrangement in Shared Memory
- [cutlass::epilogue::threadblock::PredicatedTileIterator](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h) - a component for loading or storing matrix fragments to Global Memory (with bounds checks)
- [cutlass::epilogue::thread::LinearCombination](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/linear_combination.h) - an element-wise function computing `alpha * AB + beta * C` to compute the final output
