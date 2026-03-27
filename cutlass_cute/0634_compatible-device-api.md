---
title: "Compatible Device API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#compatible-device-api"
---

## [Compatible Device API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#compatible-device-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#compatible-device-api "Permalink to this headline")

The entry point for CUTLASS’s Device GEMM API
is the class
`cutlass::gemm::device::GemmUniversalAdapter`.
This class lives in the header file
[include/cutlass/gemm/device/gemm_universal_adapter.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).

`GemmUniversalAdapter` is a “universal adapter”
and serves as a common device interface
for both CUTLASS 3.x and CUTLASS 2.x kernels.
Its template parameter `GemmKernel`,
the GEMM kernel type, can be any of the following:

- `cutlass::gemm::kernel::GemmUniversal`,
implementing CUTLASS 3.x API kernels;
- `cutlass::gemm::kernel::GemmUniversal`,
implementing CUTLASS 2.x API kernels;
- Any valid CUTLASS 2.x `kernel` layer GEMM that
was previously composable with `device::GemmUniversalAdapter`

Users implementing new kernels in either API should prefer
using `kernel::GemmUniversal` as the kernel type
and compose it with `device::GemmUniversalAdapter`.
Users with existing `kernel::Gemm` kernels
can continue to use them as template arguments
of `device::GemmUniversalAdapter`. They can adopt
`GemmUniversal` as a gradual migration path,
since `GemmUniversal` accepts either 3.0 or 2.x collectives.
Please see the [next section for `kernel::GemmUniversal`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#compatible-kernel-api) for details.

`GemmUniversalAdapter` presents a single
host-side interface to both 3.0 and 2.x kernels.
CUTLASS accomplishes this by
specializing `GemmUniversalAdapter`’s implementation
on either 2.x API implementing kernel layer GEMMs, or 3.x API
implementing kernel layer GEMMs (as detected by `gemm::detail::IsCutlass3GemmKernel`
discussed below). As a result, `GemmUniversalAdapter`’s behavior
might differ between the two specializations.
