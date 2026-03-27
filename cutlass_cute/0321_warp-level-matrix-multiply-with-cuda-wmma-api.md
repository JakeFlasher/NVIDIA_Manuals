---
title: "Warp-level Matrix Multiply with CUDA WMMA API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#warp-level-matrix-multiply-with-cuda-wmma-api"
---

## [Warp-level Matrix Multiply with CUDA WMMA API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#warp-level-matrix-multiply-with-cuda-wmma-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#warp-level-matrix-multiply-with-cuda-wmma-api "Permalink to this headline")

The following table summarizes supported warp level shapes for each WmmaTensorOp instruction.

| **Opcode Class** | **Instruction Shape** | **Warp Shapes** |
| --- | --- | --- |
| **WmmaTensorOp** | 16-by-16-by-16 | 32x32x16, 32x64x16, 64x32x16 |
| **WmmaTensorOp** | 8-by-32-by-16 | 32x32x16, 32x64x16, 64x32x16 |
| **WmmaTensorOp** | 32-by-8-by-16 | 32x32x16, 32x64x16, 64x32x16 |
| **WmmaTensorOp** | 8-by-8-by-32 | 32x32x32, 32x64x32, 64x32x32, 64x64x32 |
| **WmmaTensorOp** | 8-by-8-by-128 | 32x32x128, 32x64x128, 64x32x128, 64x64x128 |

CUDA exposes warp-level matrix operations in the CUDA C++ WMMA API. The CUDA C++ WMMA API exposes Tensor Cores via a set of functions and types in the `nvcuda::wmma` namespace. The functions and types in `nvcuda::wmma` provide target-independent APIs and implement architecture-specific tensor operation using TensorOp instruction underneath. CUTLASS exposes WMMA API through WmmaTensorOp. The WmmaTensorOp supports canonical shared memory layouts. The following table summarizes the destination shared memory layout that can be targeted by matrix operands. The WMMA API expects that matrices in shared memory loaded by `nvcuda::wmma::load_matrix_sync()` satisfy 128 bit alignment.

**WmmaTensorOp (all matrix sizes and data types).**

| **Operand** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- |
| **A** | `RowMajor`, `ColumnMajor` | `RowMajor`, `ColumnMajor` |
| **B** | `RowMajor`, `ColumnMajor` | `RowMajor`, `ColumnMajor` |
| **C** | `RowMajor`, `ColumnMajor` | `RowMajor`, `ColumnMajor` |
