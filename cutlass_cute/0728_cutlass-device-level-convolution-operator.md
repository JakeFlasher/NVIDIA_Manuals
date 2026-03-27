---
title: "CUTLASS Device-level Convolution Operator"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#cutlass-device-level-convolution-operator"
---

# [CUTLASS Device-level Convolution Operator](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-device-level-convolution-operator)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-device-level-convolution-operator "Permalink to this headline")

CUTLASS defines CUDA C++ templates accepting numerous template arguments to specialize the resulting
kernel by operation, data type, tile configuration, math instruction, and fused output operation.

In [turing_tensorop_conv2dfprop.cu](https://github.com/NVIDIA/cutlass/tree/main/examples/09_turing_tensorop_conv2dfprop/turing_tensorop_conv2dfprop.cu), a convolution
operation is defined as follows.

```c++
/// Define an Implicit GEMM convolution forward propagation (fprop) kernel
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA,                                          // data type of element a (mapped to activation for fprop)
  LayoutInputA,                                           // layout of element a (mapped to activation for fprop)
  ElementInputB,                                          // data type of element b (mapped to filters for fprop)
  LayoutInputB,                                           // layout of element b (mapped to filters for fprop)
  ElementC,                                               // data type of element c (mapped to output for fprop)
  LayoutC,                                                // layout of element c (mapped to output for fprop)
  ElementAccumulator,                                     // data type of internal accumulation
  MMAOp,                                                  // opcode class tag
  SmArch,                                                 // target SM architecture
  ThreadblockShape,                                       // shape of threadblock tile
  WarpShape,                                              // shape of warp-level GEMM tile
  InstructionShape,                                       // shape of target math instruction
  EpilogueOp,                                             // epilogue operator
  SwizzleThreadBlock,                                     // optional function to reorder threadblocks for locality
  NumStages,                                              // number of pipeline stages in threadblock-scoped GEMM
  cutlass::arch::OpMultiplyAddSaturate,                   // math operation on data of element a and b
  cutlass::conv::IteratorAlgorithm::kOptimized            // global memory iterator algorithm
>::Kernel
```

This template is intended to be generic and cover all feasible configurations. The example specifies
the following concrete data types, layouts, and tile shapes.

```c++
/// Define an Implicit GEMM convolution forward propagation (fprop) kernel
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  cutlass::int4b_t,                                    // data type of element a (mapped to activation for fprop)
  cutlass::layout::TensorNHWC,                         // layout of element a (mapped to activation for fprop)
  cutlass::int4b_t,                                    // data type of element b (mapped to filters for fprop)
  cutlass::layout::TensorNHWC,                         // layout of element b (mapped to filters for fprop)
  int32_t,                                             // data type of element c (mapped to output for fprop)
  cutlass::layout::TensorNHWC,                         // layout of element c (mapped to output for fprop)
  int32_t,                                             // data type of internal accumulation
  cutlass::arch::OpClassTensorOp,                      // opcode class tag
  cutlass::arch::Sm75,                                 // target SM architecture
  cutlass::gemm::GemmShape<128, 128, 128>,             // shape of threadblock tile
  cutlass::gemm::GemmShape<64, 64, 128>,               // shape of warp-level GEMM tile
  cutlass::gemm::GemmShape<8, 8, 32>,                  // shape of target math instruction
  cutlass::epilogue::thread::LinearCombinationClamp<
    int32_t,                                           // data type of output matrix
    8,                                                 // The number of elements per vectorized
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    int32_t,                                           // Data type of accumulator
    float>;    ,                                       // epilogue operator
  SwizzleThreadBlock,                                  // optional function to reorder threadblocks for locality
  2,                                                   // number of pipeline stages in threadblock-scoped GEMM
  cutlass::arch::OpMultiplyAddSaturate,                // math operation on data of element a and b
  cutlass::conv::IteratorAlgorithm::kOptimized         // global memory iterator algorithm
>::Kernel
```

That is, this computes 2D convolutional forward propagation with 4-bit integer inputs and outputs (`cutlass::int4b_t`).
Internal accumulation is performed using 32-bit integers (`int32_t`), and an elementwise linear combination operation
is performed on the output in single-precision floating point (`float`).

The threadblock and warp-level tile shapes refer to the hierarchically blocked GEMM computation
[described here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html). Larger tiles achieve greater reuse of data loaded through shared memory
but launch fewer CTAs and may not fully occupy the GPU for small problem sizes. Smaller tile configurations achieve
lower peak utilizations but may better match the number of SMs within the GPU for real-world workloads.
