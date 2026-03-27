---
title: "Emitting kernels via emit_kernel_listing.py"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#emitting-kernels-via-emit-kernel-listing-py"
---

# [Emitting kernels via emit_kernel_listing.py](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#emitting-kernels-via-emit-kernel-listing-py)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#emitting-kernels-via-emit-kernel-listing-py "Permalink to this headline")

We provide a Python script `emit_kernel_listing.py` that allows a user to selectively test a subset of profiler-based kernels stamped out in `generator.py`. A unique benefit to generate kernels and test via this script is that it can feed a series of runtime arguments, such as different `M`/`N`/`K` and `alpha`/`beta`, to each kernel, instead of relying on a single default value. It also properly generates runtime datatype and cluster shapes for certain kernels to help reduce the generated kernel count and accordingly the total compilation time. An interested user may refer to [emit_kernel_listing.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/emit_kernel_listing.py) for details. To enable this new feature, a user should add `-DCUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=ON` when building CUTLASS profiler.
