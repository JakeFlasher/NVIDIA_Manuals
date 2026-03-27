---
title: "CUTLASS 2.x Kernels"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#cutlass-2-x-kernels"
---

### [CUTLASS 2.x Kernels](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-2-x-kernels)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-2-x-kernels "Permalink to this headline")

| **Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type** | **Layouts** | **Unit Test** |
| --- | --- | --- | --- | --- | --- |
| **Simt** | 50+ | 11.4+ | `f32 * f32 + f32 => f32` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_sgemm_nt_sm50.cu) |
| **Simt** | 50+ | 11.4+ | `f64 * f64 + f64 => f64` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_dgemm_nt_sm50.cu) |
| **Simt** | 60+ | 11.4+ | `f16 * f16 + f16 => f16` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_hgemm_nt_sm50.cu) |
| **Simt** | 61+ | 11.4+ | `s8 * s8 + s32 => {s32,s8}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_igemm_nt_sm50.cu) |
| **WmmaTensorOp** | 70+ | 11.4+ | `f16 * f16 + f16 => f16` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16t_f16n_wmma_tensor_op_f16_sm70.cu) |
| **WmmaTensorOp** | 70+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16t_f16n_wmma_tensor_op_f32_sm70.cu) |
| **WmmaTensorOp** | 75+ | 11.4+ | `s8 * s8 + s32 => {s32, s8}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_wmma_tensor_op_s32_sm72.cu) |
| **WmmaTensorOp** | 75+ | 11.4+ | `s4 * s4 + s32 => {s32, s4}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_s4t_s4n_s32t_wmma_tensor_op_s32_sm75.cu) |
| **WmmaTensorOp** | 75+ | 11.4+ | `b1 ^ b1 + s32 => {s32, b1}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_b1t_b1n_s32n_wmma_tensor_op_s32_sm75.cu) |
| **TensorOp** | 70+ | 11.4+ | `f16 * f16 + f16 => f16` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_volta_tensor_op_f16_sm70.cu) |
| **TensorOp** | 70+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f32t_volta_tensor_op_f32_sm70.cu) |
| **TensorOp** | 75+ | 11.4+ | `f16 * f16 + f16 => f16` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f16_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_f16n_f16t_f32t_tensor_op_f32_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `s8 * s8 + s32 => {s32, s8}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `s4 * s4 + s32 => {s32, s4}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_s4t_s4n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `b1 ^ b1 + s32 => {s32, b1}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_b1t_b1n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp** | 80+ | 11.4+ | `f16 * f16 + f16 => f16` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f16_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `bf16 * bf16 + f32 => {bf16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_bf16t_bf16t_bf16t_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `tf32 * tf32 + f32 => f32` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_tf32n_tf32t_f32t_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `s8 * s8 + s32 => {s32, s8}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `s4 * s4 + s32 => {s32, s4}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `b1 ^ b1 + s32 => {s32, b1}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_b1t_b1n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `f64 * f64 + f64 => f64` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `cf32 * cf32 + cf32 => cf32` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf32n_cf32t_cf32t_tensor_op_tf32_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `cf64 * cf64 + cf64 => cf64` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cu), [Gaussian 3m](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cu) |
| **SpTensorOp** | 80+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp** | 80+ | 11.4+ | `bf16 * bf16 + f32 => {bf16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp** | 80+ | 11.4+ | `tf32 * tf32 + f32 => f32` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32n_f32n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp** | 80+ | 11.4+ | `s8 * s8 + s32 => {s8, s32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu) |
| **SpTensorOp** | 80+ | 11.4+ | `s4 * s4 + s32 => {s4, s32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu) |
| **TensorOp** | 90+ | 11.8+ | `f64 * f64 + f64 => f64` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu) |
