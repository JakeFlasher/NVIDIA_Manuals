---
title: "Building a Block Scaled Kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#building-a-block-scaled-kernel"
---

## [Building a Block Scaled Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#building-a-block-scaled-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#building-a-block-scaled-kernel "Permalink to this headline")

For non-blockscaled dense GEMM refer to [quick start page](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#instantiating-a-blackwell-sm100-gemm-kernel). An example dense GEMM can be found:

1. [Blackwell FP16 GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm/).

An example sparse GEMM can be found:

1. [Blackwell FP16 Sparse GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/83_blackwell_sparse_gemm/).

Narrow precision and block scaled narrow precision kernels can be built using CUTLASS 3.x collective builder interface
(as described in [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#cutlass-3-0-gemm-api)). However, special attention needs to be given to
A and B matrix layouts, alignment requirements, and dispatch policies to obtain a functionally correct and performant kernel
which are listed above.

Several examples of block scaled dense GEMM kernels can be found in [examples/72_blackwell_narrow_precision_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/) directory:

1. [NVF4 Gemm with block scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)
2. [NVF4 Gemm with block scaling and NVF4 output matrix](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
3. [Mixed precision Mxf8 x Mxf8 GEMM with block scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu)

Several examples of block scaled sparse GEMM kernels can be found in [examples/84_blackwell_narrow_precision_sparse_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm) directory:

1. [NVF4 Gemm with block scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84a_blackwell_nvfp4_bf16_sparse_gemm.cu)
2. [Mixed precision Mxf4 x Mxf8 GEMM with block scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84b_blackwell_mixed_mxfp8_bf16_sparse_gemm.cu)

Collective builder interface expects the same arguments as any other CUTLASS 3.x kernels as described
[here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#collective-builder-for-collectivemmas) with a small difference for Collective MMA builder interface.
As in all Blackwell kernels, the `TileShape_MNK` argument expects the `MmaTileShape_MNK` which is the tile shape needed
by 1 or 2 SM `tcgen05.mma` instructions.

Let’s consider building a block scaled GEMM where the A matrix is of type `mx_float4_t` and column-major (N), and the
B matrix is of type `mx_float4_t` and row-major (T). We first need to describe the A and B tensors, and find the
instruction that can support the selected A and B type and layout pair. Then, we will choose the performance parameters.

The skeleton C++ code is shown below:

```cpp
  ///////////////////////////////////////////////////////////
  //                Mainloop Builder Setup
  ///////////////////////////////////////////////////////////

  ///////////////////////////////////////////
  // 1. Describe A and B tensors
  ///////////////////////////////////////////
  using ElementA       = // TBD
  constexpr int AlignA = // TBD
  using GmemLayoutA    = // TBD
  using ElementB       = // TBD
  constexpr int AlignB = // TBD
  using GmemLayoutB    = // TBD

  // Mma's accumulator type
  using ElementAccumulator = float;           // Always float for block scaled tcgen05.mma instructions

  //////////////////////////////////////////
  // 2. Choose Performance Parameters
  //////////////////////////////////////////

  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using KernelMainloopPolicy     = // TBD
  using MmaTileShape_MNK         = // TBD
  using ClusterShape_MNK         = // TBD

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelMainloopPolicy                                                  // Kernel schedule policy.
                                                                            // Auto or using targeted scheduling policy
    >::CollectiveOp;
```

From the valid type and layout combinations [Table 3](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#bs_gemm_table), we see that only **row 3** can support `mx_float4_t`x`mx_float4_t`
combination with NT layout. As a result, we need to use the `tcgen05.mma.kind:mxf8f6f4` instruction. Additionally, in order
to use `tcgen05.mma.kind:mxf8f6f4`, we see that A and B tensors both should be 128-element aligned.
Thus, we can describe A and B tensors as follows:

```cpp
  ///////////////////////////////////////////////////////////
  //                Mainloop Builder Setup
  ///////////////////////////////////////////////////////////

  ///////////////////////////////////////////
  // 1. Describe A and B tensors
  ///////////////////////////////////////////
  using ElementA       = mx_float4_t;
  constexpr int AlignA = 128;
  using GmemLayoutA    = cutlass::layout::ColumnMajor;
  using ElementB       = mx_float4_t;
  constexpr int AlignB = 128;
  using GmemLayoutB    = cutlass::layout::RowMajor;
```

Next, we need to choose the performance parameters such as `MmaTileShape_MNK`, `KernelMainloopPolicy`,
and `ClusterShape_MNK`.

`MmaTileShape_MNK` supported for `mx_float4_t`x`mx_float4_t` with `mxf8f6f4` are listed in [Table 11](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#bs_rows_3).
For NT layout, we see that 3 `MmaTileShape_MNK` are supported: `128x128x128`, and `128x256x128` with 1SM instruction;
and `256x256x128` with 2SM instruction. Let’s say, we expect to get the best performance with `256x256x128` MMA tile shape
for our GEMM problem. Then, we need to set the `KernelMainloopPolicy` to `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100`.
Now, we need to choose the `ClusterShape_MNK`. Since we have selected a 2SM mma instruction, `ClusterShape_MNK` should be
compatible and its first mode should be a multiple of 2. `ClusterShape_MNK = cute::Shape<_2, [_1|_2|_4], _1>` or
`ClusterShape_MNK = cute::Shape<_4, [_1|_2|_4], _1>` would be valid options. Let’s choose `cute::Shape<_4,_4,_1>`.
Our performance parameters looks like below:

```cpp
  //////////////////////////////////////////
  // 2. Choose Performance Parameters
  //////////////////////////////////////////

  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using KernelMainloopPolicy     = cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using MmaTileShape_MNK         = cute::Shape<_256,_256,_128>;
  using ClusterShape_MNK         = cute::Shape<_4,_4,_1>;
```

After we config the main-loop, let’s setup the epilogue.
A normal epilogue looks like below, we need to specify the output layout, datatype, alignment and PerSmTileShape_MNK, and let others to be default/auto.

PerSmTileShape_MNK should be deduced from the mainloop setup. For example, in above mainloop setup, the MmaTileShape_MNK is
256x256x128 and the KernelMainloopPolicy is 2sm policy.
It means each CTA is doing (256 / 2sm) x 256 x 128 output, so the PerSmTileShape_MNK is 128x256x128. The possible PerSmTileShape_MNK
is listed in [Table 15](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#epi_persmtileshape)

The epilogue scheduling policy is configurable, and it is common to set `cutlass::epilogue::collective::EpilogueScheduleAuto`
to allow the epilogue builder to automatically select the appropriate policy. However, it can also be explicitly defined to
use other policies based on the 1sm or 2sm MMA instruction. The available policies are listed in [Table 14](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#epi_dispatch).

```cpp
  // Describe C and D tensors
  using ElementC = cutlass::half_t;
  constexpr int AlignC = 8;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e2m1_t;
  constexpr int AlignD = 32;
  using GmemLayoutD = cutlass::layout::RowMajor;
  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      MmaTileShape_MNK, ClusterShape_MNK,                                   // MMA tile shape, and cluster shape
      cutlass::epilogue::collective::EpilogueTileAuto,                      // Epilogue subtile shape. Auto will find a suitable tile shape
      ElementAccumulator, ElementCompute,                                   // Mma instr's accumulator type and compute precision for epilogue
      ElementC, GmemLayoutC, AlignC,                                        // C tensor description
      ElementD, GmemLayoutD, AlignD,                                        // D tensor description
      cutlass::epilogue::TmaWarpSpecialized2Sm                              // Epilogue schedule policy
    >::CollectiveOp;
```

If we want to let the epilogue generate mxf4/nvf4/mxf6/mxf8 (i.e. elements + block-scalefactor), we need to setup the epilogue fusion into the builder.
First, we need to choose a SFDVectorSize indicates how many elements sharing the same block-scalefactor.
Then, we need to choose ElementSFD and GmemLayoutSFD which indicates the output datatype and which output-dim is used to generate the block-scalefactor.
Typically, GmemLayoutSFD would be same as the GmemLayoutD.

```cpp
  //
  // Construct FusionOperation
  //
  constexpr int SFDVectorSize = 16;
  // Define the fusion operation applied during epilogue
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      SFDVectorSize,
      ElementD, ElementCompute,
      ElementSFD, GmemLayoutSFD,
      ElementC
    >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      MmaTileShape_MNK, ClusterShape_MNK,                                   // MMA tile shape, and cluster shape
      cutlass::epilogue::collective::EpilogueTileAuto,                      // Epilogue subtile shape. Auto will find a suitable tile shape
      ElementAccumulator, ElementCompute,                                   // Mma instr's accumulator type and compute precision for epilogue
      ElementC, GmemLayoutC, AlignC,                                        // C tensor description
      ElementD, GmemLayoutD, AlignD,                                        // D tensor description
      cutlass::epilogue::TmaWarpSpecialized2Sm                              // Epilogue schedule policy
      FusionOperation                                                       // <================================== Pass the fusion config into epilogue builder.
    >::CollectiveOp;
```

Above example made a gentle introduction to using the fusion operations in the epilogue. For more detailed example, see
[Blackwell GEMM with collective builder](https://github.com/NVIDIA/cutlass/tree/main/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)

Note that we have first discussed the CollectiveMainloop, then the CollectiveEpilogue for clarity.
However, the CollectiveMainloop needs to know the SMEM utilization of the epilogue. Therefore, it needs to be setup before the CollectiveMainloop. See  [examples/72_blackwell_narrow_precision_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/) directory for full kernel and run setup.
