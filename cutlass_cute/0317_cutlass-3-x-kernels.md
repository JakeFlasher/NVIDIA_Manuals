---
title: "CUTLASS 3.x Kernels"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#cutlass-3-x-kernels"
---

### [CUTLASS 3.x Kernels](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-3-x-kernels)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-3-x-kernels "Permalink to this headline")

| **Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type** | **Layouts** | **Unit Test** |
| --- | --- | --- | --- | --- | --- |
| **TensorOp** | 90a | 12.0+ | `f16 * f16 + { f16, f32 } => { f16, f32 }` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized.cu) |
| **TensorOp** | 90a | 12.0+ | `bf16 * bf16 + { f16, f32 } => { bf16, f32 }` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu) |
| **TensorOp** | 90a | 12.0+ | `{f32, tf32} * {f32, tf32} + f32 => f32` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f32_f32_f32_tensor_op_f32.cu) |
| **TensorOp** | 90a | 12.0+ | `s8 * s8 + s32 => {s32, s8}` | { T } x { N } => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu) |
