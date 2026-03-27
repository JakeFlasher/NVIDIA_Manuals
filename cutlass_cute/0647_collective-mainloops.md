---
title: "Collective Mainloops"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#collective-mainloops"
---

### [Collective Mainloops](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#collective-mainloops)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-mainloops "Permalink to this headline")

The `cutlass::gemm::collective::CollectiveMma` class
is the primary interface to the collective
matrix multiply-accumulate (MMA) mainloops.
“Mainloop” refers to the “main loop” over tiles –
the “cluster tile k” loop in the pseudocode
near the top of this document.
Any looping over multiple tiles that
the algorithm might need to do would happen here.

The `CollectiveMma` class is declared in the header
[cutlass/gemm/collective/collective_mma.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_mma.hpp).

```c++
namespace cutlass::gemm::collective {

template <
  class DispatchPolicy,
  class TileShape,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA,
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma {
  static_assert(sizeof(ElementA) == 0, "Could not find a mainloop specialization.");
};

} // namespace cutlass::gemm::collective
```

- `DispatchPolicy` is the most important type for a collective, and is
[covered in more detail below](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-dispatch-policies).
- `StrideA` and `StrideB` are instances of type `cute::Stride` that represent the global memory layout of A and B tensors. These strides are required to be rank-3, representing the modes `[outer, inner, batch]`. Each of the 3 ranks can be a multi-modal hierarchical stride; this would apply if implementing a tensor contraction.
- `TiledMma` is an instance of `cute::TiledMma`.
- `GmemTiledCopyA` and `GmemTiledCopyB` are instances of `cute::TiledCopy` types. Both tiled operation types are [covered in more detail below](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tiled-mma-and-copy).
- `SmemLayoutAtomA` and `SmemLayoutAtomB` are instances of type `cute::Layout` and represent the smallest
layout that will get tiled over the entire collective’s shared memory. This layout does _not_ include the
pipeline mode, and therefore, both are expected to be rank 2 layouts of shape [`outer`, `inner`].
- `SmemCopyAtomA` and `SmemCopyAtomB` are `Copy_Atom`s to be used for moving data from shared memory
into register memory.

Notice that CUTLASS 3.0 mainloops do not accept a dedicated accumulator element type.
We obtain the accumulator type from the `typename TiledMma::ValTypeC`. Note also that
top level API’s `ElementA` and `ElementB` can differ from those of the MMA facing
`typename TiledMma::ValTypeA` and `typename TiledMma::ValTypeB`, allowing TMA or user
supplied transform operations to perform type conversions.
