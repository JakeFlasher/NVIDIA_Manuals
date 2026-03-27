---
title: "Blackwell SM100 GEMMs"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm100-gemms"
---

# [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-sm100-gemms)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-sm100-gemms "Permalink to this headline")

[**TLDR; jump to block scaled GEMM example**](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#detailed_blockscale_example)

Blackwell SM100 introduces `tcgen05.mma` instructions. `tcgen05.mma` instructions support all legacy types (`tfloat32_t`, `half_t`, `bfloat16_t`, `int8_t`, `uint8_t`) and
the new 4, 6, and 8-bits floating point datatypes with and without scale factors.
This document explains the new `tcgen05.mma` instructions supported by CUTLASS and how one can leverage CUTLASS to create
efficient SM100 GEMM kernels targeting these new mma instructions.

Blackwell SM100 has 7 new `tcgen05.mma` instructions. These instructions are 2x to 4x faster then Hopper Architecture’s WGMMA instructions.

| Ptx Instruction | Throughput | Notes |
| --- | --- | --- |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::tf32 | 2x Hopper Tf32 Tensor Core | MMA with A={tf32} x B={tf32} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::f16 | 2x Hopper Fp16 Tensor Core | MMA with A={f16} x B={f16} or A={bf16} x B={bf16}  TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::i8 | 2x Hopper I8 Tensor Core | MMA with A={i8} x B={i8} or A={u8} x B={u8}  TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::f8f6f4 | 2x Hopper Fp8 Tensor Core | Mixed precision MMA with A={f4,f6,f8} x B={f4,f6,f8} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf8f6f4.block_scale | 2x Hopper Fp8 Tensor Core | Block scaled mixed precision MMA with A={mxf4,mxf6,mxf8} x B={mxf4,mxf6,mxf8} with TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf4.block_scale | 4x Hopper Fp8 Tensor Core | Block scaled MMA with A={mxf4} x B={mxf4} with TN layouts |
| tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf4nvf4.block_scale.scale_vec_size::[2X\|4X] | 4x Hopper Fp8 Tensor Core | Block scaled MMA with A={mxf4} x B={mxf4} or A={nvf4} x B={nvf4} with TN layouts |

For more detailed information see [`tcgen05.mma` PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions).
