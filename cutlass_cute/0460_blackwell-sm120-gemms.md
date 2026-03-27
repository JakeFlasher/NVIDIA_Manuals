---
title: "Blackwell SM120 GEMMs"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms"
---

# [Blackwell SM120 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-sm120-gemms)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-sm120-gemms "Permalink to this headline")

The NVIDIA RTX 5000 Series GPUs introduce support for new narrow precision (4bit and 6bit) block-scaled and non-block-scaled tensor cores. The PTX ISA has extended the `mma` instructions to support these data formats which are 1x to 4x faster than Ada architecture’s fp8 tensor cores. For more detailed information see [`mma` PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma).

CUTLASS 4.0 has added support for these newly introduced narrow precision GEMMs. Similar to the Blackwell SM100 GEMMs, the SM120 GEMMs can be built using the collective builder interface. See examples in [examples/79_blackwell_geforce_gemm/](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#../../examples/79_blackwell_geforce_gemm/) and unit tests listed below.

The data types supported and tensor alignment requirements are the same as the Blackwell SM100 GEMMs. The scale factor layout is also the same as SM100 mentioned above. `OpClassTensorOp` is used for non-blockscaled narrow precision GEMMs and `OpClassBlockScaledTensorOp` is used for blockscaled narrow precision GEMMs.

| Ptx Instruction | Throughput | Notes | Unit Test |
| --- | --- | --- | --- |
| mma.sync.aligned.kind::f8f6f4 | 1x Ada Fp8 Tensor Core(2x for FP32 accumulator) | Mixed precision MMA with A={f4,f6,f8} x B={f4,f6,f8} TN layouts | [unit test](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#../../test/unit/gemm/device/sm120_tensorop_gemm/) |
| mma.sync.aligned.kind::mxf8f6f4.block_scale | 1x Ada Fp8 Tensor Core(2x for FP32 accumulator) | Block scaled mixed precision MMA with A={mxf4,mxf6,mxf8} x B={mxf4,mxf6,mxf8} with TN layouts | [unit test](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf6_mxf8_f32_f32.cu) |
| mma.sync.aligned.kind::mxf4.block_scale | 2x Ada Fp8 Tensor Core(4x for FP32 accumulator) | Block scaled MMA with A={mxf4} x B={mxf4} with TN layouts | [unit test](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32.cu) |
| mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::[2X\|4X] | 2x Ada Fp8 Tensor Core(4x for FP32 accumulator) | Block scaled MMA with A={mxf4} x B={mxf4} or A={nvf4} x B={nvf4} with TN layouts | [unit test](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu) |

Besides the similarities, there are some key differences from the Blackwell SM100 GEMMs:
