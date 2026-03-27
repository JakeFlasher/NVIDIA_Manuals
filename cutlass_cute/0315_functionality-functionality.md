---
title: "Functionality"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html#functionality--functionality"
---

# [Functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#functionality)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#functionality "Permalink to this headline")

Note : CUTLASS-3 requires users to use CUDA 11.4 or newer, and SM70 or newer, for the target toolkit and architecture, respectively.

- N - Column Major Matrix
- T - Row Major matrix
- {N,T} x {N,T} - All combinations, i.e., NN, NT, TN, TT
- [NHWC](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/tensor.h#L63-206) - 4 dimension tensor used for convolution
- [NCxHWx](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/tensor.h#L290-395) - Interleaved 4 dimension tensor used for convolution
- f - floating point
- s - signed int
- b - bit
- cf - complex float
- bf16 - bfloat16
- tf32 - tfloat32
- Simt - Use Simt CUDA Core MMA
- TensorOp - Use Tensor Core MMA
- SpTensorOp - Use Sparse Tensor Core MMA
- WmmaTensorOp - Use WMMA abstraction to use Tensor Core MMA
