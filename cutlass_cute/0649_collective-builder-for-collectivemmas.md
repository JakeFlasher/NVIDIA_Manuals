---
title: "Collective Builder for CollectiveMmas"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#collective-builder-for-collectivemmas"
---

### [Collective Builder for CollectiveMmas](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#collective-builder-for-collectivemmas)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-builder-for-collectivemmas "Permalink to this headline")

The primary `CollectiveMma` is intended to be an expert user interface that allows full control over
all the properties of the collective’s GPU micro-kernel. However, often a user just wants an
off-the-shelf GEMM mainloop implementation parameterized on simple configuration parameters. CUTLASS 3.0
provides [`cutlass::gemm::collective::CollectiveBuilder`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_builder.hpp) for such scenarios.

```c++
namespace cutlass::gemm::collective {
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};
} // namespace cutlass::gemm::collective
```

`CollectiveBuilder` accepts CUTLASS 2.x equivalent input template arguments, and attempts to build
the best performing `CollectiveMma` from the given parameters.

- `ArchTag` is one of the SM architectures tags from `cutlass::arch::Sm*`.
- `OpClass` is one of the operator class tags from `cutlass::arch::OpClass*`.
- `ElementA` and `ElementB` are the logical value types of the A resp. B tensors.
- `ElementAccumulator` is the accumulator type to be used in the instruction.
- `GmemLayoutA` and `GmemLayoutB` are CUTLASS 2.x layout tags, `layout::RowMajor` or `layout::ColumnMajor`.
- `AlignmentA` and `AlignmentB` are global memory alignments of A and B tensors in terms of element count.
- `TileShape_MNK` is an instance of `cute::Shape` that is rank-3, representing the MxNxK collective tile shape.
- `ClusterShape_MNK` is an instance of `cute::Shape` that is rank-3, representing the MxNxK threadblock cluster tile shape.
- `StageCountType` is either `collective::StageCountAuto` or an instance of `collective::StageCount<N>`.
- `KernelScheduleType` is either `collective::KernelScheduleAuto` or one of the specific kernel schedule tags discussed in the [dispatch policy section](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-dispatch-policies) above.

`StageCountAuto` allows the collective builder to compute the size of a single stage’s size in shared memory
and maximize the shared memory usage assuming 1 threadblock / multiprocessor occupancy.

`KernelScheduleAuto` allows the collective builder to pick the best kernel schedule available for the
given set of parameters, or let’s the user override this with a specific kernel schedule type.

Note that collective builders are still in beta, and their functionality
does not map onto the full design space that the primary expert `CollectiveMma` API
allows for. We expect their supported mainloop types to expand in future releases, but
with 3.0, only SM90 tensorop kernels are supported through the builder API. The builder API
may also change in the future as we adopt user feedback.

If the builder is able to provide a collective mainloop type for the given set of parameters,
it will be aliased within as `CollectiveOp`. For more information on how to
parameterize kernels conveniently with the collective builder, please see example [49_hopper_gemm_with_collective_builder](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder).
