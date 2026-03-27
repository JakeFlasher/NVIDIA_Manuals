---
title: "Kernel API design differences"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#kernel-api-design-differences"
---

### [Kernel API design differences](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#kernel-api-design-differences)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#kernel-api-design-differences "Permalink to this headline")

The CUTLASS 2.x Kernel API was more closely tied
to the Device API, as we mentioned above.
In particular, the 2.x Device API specified the grid shape
used to launch the Kernel API.
In CUTLASS 3.0, the Kernel API controls its own grid shape,
while the device adapter simply queries the kernel with which it needs to be launched.

This change is required to support various kernel schedules
that may need their own schedule specific grid planning logic.
For example, persistent kernel schedules generally only launch with
as many threadblocks as the number of multiprocessors on the GPU.

All CUTLASS 3 `kernel::GemmUniversal` specializations expose the following (static) API:

```c++
// Returns true if the kernel can execute the provided GEMM arguments.
static bool
can_implement(Arguments const& args);

// Returns a dim3 representing the threadblock shape.
static dim3
get_block_shape();

// Returns a dim3 representing the grid shape in terms of threadblocks.
static dim3
get_grid_shape(Params const& params);
```

The device adapter simply queries the kernel for these three before launching it on the device.
CUTLASS 3.0 provides a meta-function to detect whether a `cutlass::gemm::kernel::*` implements
the 3.x API or 2.x API:

```c++
// include/cutlass/gemm/gemm.h

namespace cutlass:gemm::detail {

// The following metafunction is used to detect whether a
// `kernel::Gemm` or `kernel::GemmUniversal` implements the CUTLASS 3.x API,
// by checking whether the problem shape type is aliased within.
template <class GemmKernel, class = void>
struct IsCutlass3GemmKernel;

} // namespace cutlass:gemm::detail
```

Users can dispatch their generic code against 2.x and 3.x specializations with
this as a type trait for the kernel API version.
