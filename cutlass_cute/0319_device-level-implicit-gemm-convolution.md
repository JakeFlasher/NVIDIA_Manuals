---
title: "Device-level Implicit GEMM convolution"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#device-level-implicit-gemm-convolution"
---

## [Device-level Implicit GEMM convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#device-level-implicit-gemm-convolution)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#device-level-implicit-gemm-convolution "Permalink to this headline")

The following table summarizes device-level implicit GEMM convolution kernels in CUTLASS, organized by opcode class, data type, and layout.
Hyperlinks to relevant conv2d fprop unit tests demonstrate how specific template instances may be defined.
One can find and/or create equivalent dgrad and wgrad convolutional operators.

| **Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type** | **Layouts** | **Unit Test** |
| --- | --- | --- | --- | --- | --- |
| **Simt** | 50+ | 11.4+ | `f32 * f32 + f32 => f32` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm50.cu) |
| **Simt** | 50+ | 11.4+ | `cf32 * cf32 + cf32 => cf32` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_sm50.cu) |
| **TensorOp** | 70+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm70.cu) |
| **TensorOp** | 75+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `s8 * s8 + s32 => {s32, s8}` | NHWC, NCxHWx | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_sm75.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.cu) |
| **TensorOp** | 75+ | 11.4+ | `s4 * s4 + s32 => {s32, s4}` | NHWC, NCxHWx | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm75.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4ncxhwx_s4cxrskx_s4ncxhwx_tensor_op_s32_sm75.cu) |
| **Simt** | 80+ | 11.4+ | `f32 * f32 + f32 => f32` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm80.cu) |
| **Simt** | 80+ | 11.4+ | `cf32 * cf32 + cf32 => cf32` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `f16 * f16 + f32 => {f16, f32}` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `f16 * f16 + f16 => f16` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `tf32 * tf32 + f32 => f32` | NHWC | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `s8 * s8 + s32 => {s32, s8}` | NHWC, NCxHWx | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_sm80.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.cu) |
| **TensorOp** | 80+ | 11.4+ | `s4 * s4 + s32 => {s32, s4}` | NHWC, NCxHWx | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm80.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4ncxhwx_s4cxrskx_s4ncxhwx_tensor_op_s32_sm80.cu) |
