---
title: "Collective Dispatch Policies"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#collective-dispatch-policies"
---

### [Collective Dispatch Policies](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#collective-dispatch-policies)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#collective-dispatch-policies "Permalink to this headline")

`CollectiveMma` implementations are not generic.
Instead, they must be specialized for each algorithm and GPU architecture.
Users can dispatch to a `CollectiveMma` specialization
by picking template arguments matching that specialization.
CUTLASS 3.0 adopts a tag-based dispatch policy type to specialize
mainloop implementations and add tuning knobs to them.

Below is an example of one of the dispatch policies that is used to dispatch to a Hopper TMA
warp-specialized mainloop implementation:

```c++
// n-buffer in smem (Hopper TMA),
// pipelined with Hopper GMMA and TMA,
// warp-specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

The `Stages_` template parameter lets the user freely vary the number of pipeline stages,
while the `ClusterShape_` type allows for parameterization over the shape of the threadblock
cluster over which TMA multicast will take place.

The collective dispatch policy is also the primary point of composing various kernel schedules
freely with any mainloop. Each mainloop policy either prescribes a `Schedule` with which
it needs to be run, or exposes a template API that lets the user pick a subset of the following schedules:

```c++
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
```

- A single kernel schedule can support multiple mainloop implementations. For example,
`KernelMultistage` can be composed with many different mainloop implementations across GPU
architectures such as `MainloopSm70TwoStage`, `MainloopSm80CpAsyncUnpredicated`, and many more.
- A single mainloop can be composed with multiple
possible kernel schedules. For example, the `MainloopSm90TmaGmmaWarpSpecialized` can be
composed with any of the `KernelTmaWarpSpecialized`, `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative`
kernel schedules.

As [discussed in the CUTLASS 3.0 design documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html), adopting tag
dispatch policies for our core vocabulary types allows us to maintain a single type name for
all operations that conceptually belong to the same class. This design has the following benefits.

- It _avoids code duplication_ in cases where mainloops can be composed with multiple kernels or vice versa.
- It _makes writing generic code easier_, as the primary type name `CollectiveMma` does not change across any implementation.
- It _provides a clear, singular extension point_ for users to plug in new, custom mainloops implementations specialized on their own dispatch policies.
