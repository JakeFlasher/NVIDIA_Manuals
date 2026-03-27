---
title: "Unit Tests"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#unit-tests"
---

## [Unit Tests](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#unit-tests)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#unit-tests "Permalink to this headline")

Unit tests verify the functional behavior of each of the above components in a standalone CUDA kernel. This provides a
convenient environment to

a. inspect the template definition,
b. showcase instantiation of use of these templates in device code, and
c. assert functional correctness.

**Convolution unit tests**

- Device-wide convolution operator: [conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm75.cu](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm75.cu)

**GEMM unit tests**

- Warp-scoped matrix multiply for Turing Tensor Cores: [gemm_sm75.cu](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/warp/gemm_sm75.cu)

**Epilogue unit tests**

- Epilogue for Turing Tensor Cores: [epilogue_tensor_op.cu](https://github.com/NVIDIA/cutlass/tree/main/test/unit/epilogue/threadblock/epilogue_tensor_op.cu)
