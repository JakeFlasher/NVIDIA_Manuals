---
title: "Compatible Kernel API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#compatible-kernel-api"
---

## [Compatible Kernel API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#compatible-kernel-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#compatible-kernel-api "Permalink to this headline")

CUTLASS 3.x API shares the kernel layer API with CUTLASS 2.x
through the single entry point type `cutlass::gemm::kernel::GemmUniversal`.
All kernel layer GEMMs are viewed as a composition of a collective mainloop
and a collective epilogue.

**`kernel::GemmUniversal` implements both 2.x and 3.x APIs**

The entry point for CUTLASS’s kernel API is the class
`cutlass::gemm::kernel::GemmUniversal`.
This class’ declaration lives in the header file
[include/cutlass/gemm/kernel/gemm_universal.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.hpp).

```c++
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 * SFIANE shims both 2.x and 3.0 API kernels based on ProblemShapeOrThreadblockMma_.
**/
template <
  class ProblemShapeOrThreadblockMma_,
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
```

We call this class “universal” because it can be built
using either the CUTLASS 3.0 or the 2.x mainloops and epilogues.
If `GemmUniversal`’s first template argument
(`ProblemShapeOrThreadblockMma_`) is a `cute::tuple`,
then `GemmUniversal` assumes that
the remaining three template arguments
(the mainloop, epilogue, and grid swizzle)
implement the 3.0 APIs.
Otherwise, `GemmUniversal` assumes that
the remaining three template arguments
implement the 2.x APIs.
All the template arguments must be either
CUTLASS 3.0 or CUTLASS 2.x types. For example,
`GemmUniversal` does not permit using
a 2.x mainloop with a 3.0 collective epilogue.

CUTLASS 3.x implements various embodiments of `kernel::GemmUniversal`.
Each kernel layer schedule is specialized
for a GEMM scheduling algorithm and GPU architecture.
Specializations of `kernel::GemmUniversal` for 3.0 APIs live in
any of various `gemm_*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
The specialization to which to dispatch is decided through the dispatch policy’s `Schedule` type.

Specializations for 2.x APIs live in the header file
[include/cutlass/gemm/kernel/gemm_universal.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.h).
