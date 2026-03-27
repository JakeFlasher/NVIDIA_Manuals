---
title: "CUTLASS GEMM Components"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#cutlass-gemm-components"
---

## [CUTLASS GEMM Components](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-gemm-components)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-gemm-components "Permalink to this headline")

CUTLASS expresses the above loop nest
with the following components which are specialized for
data type, layout, and math instruction.

| API level | API Class and/or function names |
| --- | --- |
| Device | `cutlass::gemm::device::GemmUniversalAdapter` |
| Kernel | `cutlass::gemm::kernel::GemmUniversal` |
| Collective | `cutlass::gemm::collective::CollectiveMma` <br> `cutlass::epilogue::collective::DefaultEpilogue` <br> `cutlass::epilogue::collective::Epilogue` <br> |
| Tiled (MMA and Copy) | `cute::TiledMma` and `cute::TiledCopy` <br> `cute::gemm()` and `cute::copy()` |
| Atom | `cute::Mma_Atom` and `cute::Copy_Atom` |

In CUTLASS 3.0, we assemble kernels
by first composing a collective mainloop and collective epilogue
together at the kernel layer,
and then wrapping them with a host-side adapter
to form a GEMM handle to that kernel.

The following sections describe these components
in the order a user should instantiate them
in order to assemble a kernel.  This order is

1. assemble the required collective mainloop and epilogues,
2. compose them together to build a kernel type, and
3. wrap up the kernel with a device layer adapter.

This order is also reflected in the [CUTLASS 3.0 Hopper kernel examples](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) as seen in the excerpt below.

```c++
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Towards the end, we also briefly cover CuTe’s tiled mma and copy as well as the atom layer APIs,
before redirecting users to CuTe-specific documentation for further details.
