---
title: "Kernel API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#kernel-api"
---

## [Kernel API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#kernel-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#kernel-api "Permalink to this headline")

The kernel is “a collection of all clusters in the grid.”
The kernel layer schedules have four main responsibilities.

- Ordering the execution of collectives within the kernel, performing any synchronization between that may be necessary
- Marshalling the threads of a warp specialized schedules into their respective roles
- Performing any necessary grid swizzling logic
- Tiling the input tensors with the threadblock cluster value tile before invoking the collectives on them

The Kernel API is the entry point for a grid of thread blocks
that may or may not be organized in a cluster.
It is the composition point for fusing back-to-back GEMMs,
epilogues, and/or other operations.

The entry point API for CUTLASS 3.0 kernel is the class
`cutlass::gemm::kernel::GemmUniversal`, found in the header file
[include/cutlass/gemm/kernel/gemm_universal.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.hpp).
`GemmUniversal` is a stateless universal device kernel
that implements GEMM as the composition of two parts:

- a collective mainloop, and
- a collective epilogue

```cpp
namespace cutlass::gemm::kernel {
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
**/
template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
} // namespace cutlass::gemm::kernel
```

_Stateless_ means that the caller –
for example, the Device API described above –
manages the kernel’s state.
The kernel just takes input and output parameters (`Params`).

_Universal_ means that `GemmUniversal` works
for both CUTLASS 3.0 and 2.x interfaces
and across a broad range of kernel schedules.
If `GemmUniversal`’s first template argument is a `cute::Shape`,
then `GemmUniversal` assumes that the remaining template arguments
implement the 3.0 APIs.  Otherwise, `GemmUniversal` assumes that
the remaining template arguments implement the 2.x APIs.
Starting with CUTLASS 3.0, the problem shape has been promoted
to a top-level template API for the GEMM kernel.
This supports fully static GEMM instantiations
where the user expects to know some or all
of the problem shapes at compile time
in order to extract even more performance.

The _collective mainloop_ implements MMA on local tiles.
The _collective epilogue_ addresses any operations after the MMA,
such as applying the `beta * C` part of `C := beta * C + alpha * A * B`.
We will explain _collective_ in more detail below.

Specializations of `kernel::GemmUniversal` for 3.0 APIs live in
any of various `gemm_*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
Specializations for 2.x APIs can be found in the header file
[include/cutlass/gemm/kernel/gemm_universal.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.h).

CUTLASS 3.x implements various embodiments of `kernel::GemmUniversal`.
Each kernel layer schedule is specialized
for a GEMM scheduling algorithm and GPU architecture.
Specializations of `kernel::GemmUniversal` for 3.0 APIs live in
any of various `include/cutlass/gemm/kernel/{arch_tag}*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
Which specialization to dispatch to is decided through the dispatch policy’s `Schedule` type.

For example, the header file
[include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)
has a specialization of `kernel::GemmUniversal` for Hopper
that uses a warp-specialized mainloop with a persistent scheduling algorithm,
while the header file
[include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp)
has a specialization of `GemmUniversal` for Hopper
that uses a warp-specialized but non-persistent algorithm.

To support composition between supported kernel schedules and mainloop dispatch policies without having to
duplicate collective mainloop implementations, GEMM kernel layer schedules can be composed with
any mainloop that specifies their corresponding kernel schedule as their `Schedule` type in the policy.
This is discussed in detail in the [collective dispatch policy section](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-dispatch-policies) above.

```c++
// An example of the SM90 KernelMultistage kernel's
// specialization logic that allows it to be composed
// with many mainloops such as `MainloopSm80CpAsync`
// and `MainloopSm70TwoStage`.
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  std::enable_if_t<std::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```
