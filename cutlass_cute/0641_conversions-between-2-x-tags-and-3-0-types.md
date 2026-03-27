---
title: "Conversions between 2.x tags and 3.0 types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#conversions-between-2-x-tags-and-3-0-types"
---

### [Conversions between 2.x tags and 3.0 types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#conversions-between-2-x-tags-and-3-0-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#conversions-between-2-x-tags-and-3-0-types "Permalink to this headline")

Starting with CUTLASS 3.0, all layouts are described using
`cute::Shape` and `cute::Stride` which compose into a `cute::Layout<Shape, Stride>`.
In CUTLASS 2.x, various layout tags such as `cutlass::layout::RowMajor` are used to specialize
template implementations. These tag types only encode information about the tensor strides,
as 2.x layouts did not incorporate any concept of tensor shape in the layout tags themselves.
Users may find a need to convert between CUTLASS 2.x layout tags, and 3.0
CuTe stride types. CUTLASS 3.0 `gemm::collective::CollectiveBuilder` interfaces
also accept these 2.x layout tags as input parameters in their template API as a convenience for users.
At every entry point into CUTLASS 3.0, these tags get converted to their corresponding CuTe Stride type with
metafunctions that best approximate their corresponding `cute::Stride`.

- `cutlass::gemm::detail::TagToStrideA_t<LayoutTag>`
- `cutlass::gemm::detail::TagToStrideB_t<LayoutTag>`
- `cutlass::gemm::detail::TagToStrideC_t<LayoutTag>`

By convention, and to match user expectations, the `cute::Stride` types that these
map onto always contain one static mode corresponding to the layout tag, and two 64-bit
dynamic stride modes corresponding to the minor mode and the batch mode. Batch
mode is included by default as all CUTLASS 3.0 kernels support packed batch-mode GEMMs
out of the box.

The [`cutlass/gemm/gemm.h#440`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/gemm.h#440)
header file includes functions
that can be useful for converting
from CUTLASS 3.0 `cute::Stride`s back to CUTLASS 2.x layout tags.

- `cutlass::gemm::detail::StrideToLayoutTagA_t<CuteStride>`
- `cutlass::gemm::detail::StrideToLayoutTagB_t<CuteStride>`
- `cutlass::gemm::detail::StrideToLayoutTagC_t<CuteStride>`

These metafunctions take the CuTe Stride as a template parameter and
attempt to find the size-1 stride in the idiomatic M, N, or K modes
to best approximate a corresponding 2.x layout tag type.
Note that this may not work in general for any `cute::Stride`
as the mapping between the stride and tag type is not bijective.

These mapping utilities are kept in a `detail` namespace
as we do not guarantee stability of their implementation.
Their behavior may change in future releases as we add new features.
However, we do expect these type names to remain stable. For users who want
these 2.x reflective types from an assembled kernel with a more stable API,
the specialization of `cutlass::gemm::device::GemmUniversalAdapter`
for CUTLASS 3.0 kernel provides all aliases for all 2.x type aliases
in addition to the layout tags. You can see how they are used in the header file
[`cutlass/gemm/device/gemm_universal_adapter.h`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).
Here is an excerpt.

```c++
  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = cutlass::arch::OpMultiplyAdd;

  // If our TiledMMA's instruction thread layout size is larger than 1,
  // we know it's a tensorop
  using OperatorClass = std::conditional_t<
      (cute::size(typename GemmKernel::TiledMma::AtomThrID{}) > 1),
      cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // We get the instruction shape directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  static int constexpr kStages = CollectiveMainloop::DispatchPolicy::Stages;
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x,
  // but we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M,
  // and the rest along N, with none along K.  We also always round up
  // the warp count to 4 if the tiled mma is smaller than 128 threads.
  static constexpr int WarpsInMma = std::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = gemm::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA>();
  static int constexpr kAlignmentB = gemm::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB>();
```

CUTLASS’s library and profiler use these reflective interfaces to
obtain the kernel’s configuration parameters. Users can use these to approximate the CUTLASS 2.x types
for 3.0 API kernels.  However, the reflective interfaces cannot always match the types exactly,
as the mappings are not always bijective.
