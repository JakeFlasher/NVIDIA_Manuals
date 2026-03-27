---
title: "Instantiating a Blackwell SM100 GEMM kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#instantiating-a-blackwell-sm100-gemm-kernel"
---

## [Instantiating a Blackwell SM100 GEMM kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#instantiating-a-blackwell-sm100-gemm-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#instantiating-a-blackwell-sm100-gemm-kernel "Permalink to this headline")

Blackwell SM100 kernels are instantiated very similarly to Hopper kernels. Let us start with an
[FP8 GEMM without blockscaling](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_gemm_f8_f8_f8_tensor_op_s32_batch_alpha_beta.cu)
as an example.

The kernel starts with setting up datatypes and cluster shapes.

```c++
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::float_e4m3_t;
  using ElementD = cutlass::float_e4m3_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::half_t;
  using MmaTileShape = cute::Shape<_128,_64,Int<128 / sizeof(ElementA)>>;
  using ClusterShape = cute::Shape<_1,_1,_1>;
```

The epilogue needs to be instantiated first as the mainloop collective builder takes the shared memory budget of epilogue in the template parameter list. The 3.x epilogue collective builder API has not changed
for Blackwell, so the epilogue fusion is built in a same way as an SM90 epilogue.

```c++
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

  using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,
    ElementCompute,
    ElementC
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementD, LayoutC, 16 / sizeof(ElementD),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;
```

One can refer to our Sm100 unit tests as examples of how to correctly
choose mainloop schedules. All of our dispatch policies can be found in [dispatch_policy.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp)
and more comprehensive Blackwell specific documentation for valid
dispatch policies can be in [blackwell_functionality.md](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).

```c++
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;
```

Instantiating a blockscaled GEMM kernel is slightly different. Referring to an [MXFP8 GEMM](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_gemm_mxf8_mxf8_mxf8_tensor_op_f32_auto.cu) sample unit test, it takes a different tensor operation class:

```c++
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
```

are needed in the mainloop builder:

```c++
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleAuto
    >::CollectiveOp;
```

We encourage a user to refer to Sm100 unit tests and the generated profiler-based kernels as more comprehensive samples.
