---
title: "Device API design differences"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#device-api-design-differences"
---

### [Device API design differences](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#device-api-design-differences)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#device-api-design-differences "Permalink to this headline")

In CUTLASS 2.x, the Device API was more closely tied
to the Kernel API.  In CUTLASS 3.0, the Device API
accepts any kernel type that meets the Kernel API
interface requirements.  CUTLASS 3.0’s Device API code is
parameterized by the kernel type, but this code
is _generic_; the same code works for any kernel type.

The device layer compatibility interface, `device::GemmUniversalAdapter`,
also provides reflective mappings from 3.0-specific types
back to the closest possible 2.x equivalent types. This is [discussed further in the section below](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#conversions-between-2-x-tags-and-3-0-types).

CUTLASS 3.0’s `device::GemmUniversalAdapter` also exposes some new APIs that the 2.x `device::GemmUniversalAdapter` implementation does not. Most notably, this includes the ability to bypass the `GemmKernel::Arguments` to `GemmKernel::Params` lowering.

```c++
// Primary run() entry point API that is static allowing users to create and manage their own params.
static Status
run(Params& params, cudaStream_t stream = nullptr);
```

This new API is useful for the following scenarios.

- Running again does not require reinvoking `GemmKernel::to_underlying_arguments()`
- Manual control over construction of `GemmKernel::Params` for custom kernels with custom stride types
- Fully static problem shapes and strides for bespoke kernels where no argument mapping needs to take place
