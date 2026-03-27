---
title: "Device API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#device-api"
---

## [Device API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#device-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#device-api "Permalink to this headline")

The Device API is a universal, kernel-agnostic host interface
for kernel launch and managing the lifetime of
reusable host-side parameters.

This API is how users’ host-side .cu code
invokes CUTLASS’s single-GPU GEMM kernels.
It serves the same purpose as cuBLAS and behaves similarly.

The entry point for the Device GEMM API is the class
`cutlass::gemm::device::GemmUniversalAdapter`.
This class lives in the header file
[include/cutlass/gemm/device/gemm_universal_adapter.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).
`GemmUniversalAdapter` is a stateful, reusable handle,
which is parameterized on the `cutlass::gemm::kernel` type.

```c++
/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::*

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behavior might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;
```

_Stateful_ means that the handle instance contains state
that the kernel needs to run.
This means that the user must initialize the handle first,
then use the initialized handle instance to run the kernel.
Statefulness also means that the handle can manage the lifetime
of the kernel’s `Params` – the parameters of the kernel itself.
An important duty of `GemmUniversalAdapter`
is to map from the user’s `Arguments` –
what the user sees as the kernel’s parameters –
to the `Params` that the kernel actually sees.
For power users, the class exposes new static methods
in 3.0 APIs that can bypass stateful methods
or go directly to `Params` without intermediate `Arguments`.

_Reusable_ means that the handle instance can be used
to call the kernel multiple times with different arguments
(e.g., different matrices).
Reusing the handle may be more efficient than just
creating a new handle for each kernel invocation.

_Parameterized on the kernel type_ means that
the `GemmUniversalAdapter` class’ behavior
depends on the GEMM kernel type (see the next section).
Specifically, `GemmUniversalAdapter` has a template parameter
`GemmKernel`, which is the GEMM kernel type.
Valid template arguments for `GemmKernel` are

- `cutlass::gemm::kernel::GemmUniversal`,
implementing CUTLASS 3.x API kernels;
- `cutlass::gemm::kernel::GemmUniversal`,
implementing CUTLASS 2.x API kernels; or
- Any valid CUTLASS 2.x `kernel` layer GEMM that
was previously composable with the `device::GemmUniversalAdapter`.

`GemmUniversalAdapter` presents a single
host-side interface to both 3.0 and 2.x kernels.
CUTLASS accomplishes this by
specializing `GemmUniversalAdapter`’s implementation
on either the 2.x API implementing kernel layer GEMMs, or on the 3.x API
implementing kernel layer GEMMs. The metafunction [`cutlass::gemm::detail::IsCutlass3GemmKernel`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#kernel-api-design-differences)
is what `GemmUniversalAdapter` uses to distinguish between 2.x and 3.x kernels.

`GemmUniversalAdapter` sets up and launches the kernel, using the
CUDA extended launch API for threadblock cluster support if required.
Note, `GemmUniversalAdapter` does _not_ specify the grid shape.
The kernel controls the grid shape
and other kernel-specific launch parameters.
This makes it possible for all 3.0 kernels
to use the same kernel launch code,
thus factoring out kernel launch from the actual kernel.
