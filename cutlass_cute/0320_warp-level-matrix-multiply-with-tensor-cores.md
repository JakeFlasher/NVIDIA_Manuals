---
title: "Warp-level Matrix Multiply with Tensor Cores"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#warp-level-matrix-multiply-with-tensor-cores"
---

## [Warp-level Matrix Multiply with Tensor Cores](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#warp-level-matrix-multiply-with-tensor-cores)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#warp-level-matrix-multiply-with-tensor-cores "Permalink to this headline")

The following table summarizes supported warp level shapes for each TensorOp instruction.

| **Opcode Class** | **Instruction Shape** | **Warp Shapes** |
| --- | --- | --- |
| **TensorOp** | 8-by-8-by-4 | 32x32x4, 32x64x4, 64x32x4, 64x64x4 |
| **TensorOp** | 16-by-8-by-8 | 32x32x8, 32x64x8, 64x32x8, 64x64x8 |
| **TensorOp** | 16-by-8-by-16 | 32x32x16, 32x64x16, 64x32x16, 64x64x16 |
| **TensorOp** | 8-by-8-by-16 | 32x32x16, 32x64x16, 64x32x16, 64x64x16 |
| **TensorOp** | 8-by-8-by-32 | 32x32x32, 32x64x32, 64x32x32, 64x64x32 |
| **TensorOp** | 16-by-8-by-32 | 32x32x32, 32x64x32, 64x32x32, 64x64x32 |
| **TensorOp** | 16-by-8-by-64 | 32x32x64, 32x64x64, 64x32x64, 64x64x64 |
| **TensorOp** | 8-by-8-by-128 | 32x32x128, 32x64x128, 64x32x128, 64x64x128 |
| **TensorOp** | 16-by-8-by-256 | 32x32x256, 32x64x256, 64x32x256, 64x64x256 |
| **SpTensorOp** | 16-by-8-by-16 | 64x64x16, 64x32x16, 32x64x16, 32x32x16 |
| **SpTensorOp** | 16-by-8-by-32 | 64x64x32, 64x32x32, 32x64x32, 32x32x32 |
| **SpTensorOp** | 16-by-8-by-64 | 64x64x64, 64x32x64, 32x64x64, 32x32x64 |
| **SpTensorOp** | 16-by-8-by-128 | 64x64x128, 64x32x128, 32x64x128, 32x32x128 |

TensorOp instructions depend on a permuted shared memory layout that can be efficiently
loaded from. The following tables summarize the destination shared memory layout that
can be targeted by matrix operands. It is assumed that each thread loads 128b vectors
from global memory with layout specified in the column “GMEM Layout.”

**TensorOp 8-by-8-by-4.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `half_t` | `ColumnMajor` | `ColumnMajorVoltaTensorOpCongruous<16>` |
| **A** | `half_t` | `RowMajor` | `RowMajorVoltaTensorOpCrosswise<16>` |
| **B** | `half_t` | `ColumnMajor` | `ColumnMajorVoltaTensorOpCrosswise<16>` |
| **B** | `half_t` | `RowMajor` | `RowMajorVoltaTensorOpCongruous<16>` |
| **C** | `half_t` | `RowMajor` | `RowMajor` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**TensorOp 16-by-8-by-8.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `half_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<16>` |
| **A** | `half_t` | `RowMajor` | `RowMajorTensorOpCrosswise<16>` |
| **B** | `half_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<16>` |
| **B** | `half_t` | `RowMajor` | `RowMajorTensorOpCongruous<16>` |
| **C** | `half_t` | `RowMajor` | `RowMajor` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**TensorOp 16-by-8-by-8.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `tfloat32_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<32>` |
| **A** | `tfloat32_t` | `RowMajor` | `RowMajorTensorOpCrosswise<32>` |
| **B** | `tfloat32_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<32>` |
| **B** | `tfloat32_t` | `RowMajor` | `RowMajorTensorOpCongruous<32>` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**TensorOp 16-by-8-by-16.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `half_t`, `bfloat16_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<16>` |
| **A** | `half_t`, `bfloat16_t` | `RowMajor` | `RowMajorTensorOpCrosswise<16>` |
| **B** | `half_t`, `bfloat16_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<16>` |
| **B** | `half_t`, `bfloat16_t` | `RowMajor` | `RowMajorTensorOpCongruous<16>` |
| **C** | `half_t` | `RowMajor` | `RowMajor` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**TensorOp 8-by-8-by-4.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `double` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<64>` |
| **A** | `double` | `RowMajor` | `RowMajorTensorOpCrosswise<64>` |
| **B** | `double` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<64>` |
| **B** | `double` | `RowMajor` | `RowMajorTensorOpCongruous<64>` |
| **C** | `double` | `RowMajor` | `RowMajor` |

**TensorOp 8-by-8-by-16.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int8_t` | `RowMajor` | `RowMajorTensorOpCrosswise<8>` |
| **B** | `int8_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<8>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**TensorOp 16-by-8-by-32.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int8_t` | `RowMajor` | `RowMajorTensorOpCrosswise<8>` |
| **B** | `int8_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<8>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**TensorOp 8-by-8-by-32.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int4b_t` | `RowMajor` | `RowMajorTensorOpCrosswise<4>` |
| **B** | `int4b_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<4>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**TensorOp 16-by-8-by-64.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int4b_t` | `RowMajor` | `RowMajorTensorOpCrosswise<4>` |
| **B** | `int4b_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<4>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**TensorOp 8-by-8-by-128.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `bin1_t` | `RowMajor` | `RowMajorTensorOpCrosswise<4>` |
| **B** | `bin1_t` | `ColumnMajor` | `ColumnMajorTensorOpCongruous<4>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**SpTensorOp 16-by-8-by-16.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `tfloat32_t` | `RowMajor` | `RowMajorTensorOpCrosswise<32, 32>` |
| **B** | `tfloat32_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<32, 32>` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**SpTensorOp 16-by-8-by-32.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `half_t` | `RowMajor` | `RowMajorTensorOpCrosswise<16, 64>` |
| **B** | `half_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<16, 64>` |
| **C** | `float` | `RowMajor` | `RowMajor` |

**SpTensorOp 16-by-8-by-64.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int8_t` | `RowMajor` | `RowMajorTensorOpCrosswise<8, 128>` |
| **B** | `int8_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<8, 128>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |

**SpTensorOp 16-by-8-by-128.**

| **Operand** | **Element** | **GMEM Layout** | **SMEM Layout** |
| --- | --- | --- | --- |
| **A** | `int4b_t` | `RowMajor` | `RowMajorTensorOpCrosswise<4, 256>` |
| **B** | `int4b_t` | `ColumnMajor` | `ColumnMajorTensorOpCrosswise<4, 256>` |
| **C** | `int32_t` | `RowMajor` | `RowMajor` |
