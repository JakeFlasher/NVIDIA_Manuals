---
title: "Using dependent launch in CUTLASS"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html#using-dependent-launch-in-cutlass"
---

## [Using dependent launch in CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#using-dependent-launch-in-cutlass)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#using-dependent-launch-in-cutlass "Permalink to this headline")

When building CUTLASS, you can use the `CUTLASS_ENABLE_GDC_FOR_SM90` and `CUTLASS_ENABLE_GDC_FOR_SM100` macro
respectively to enable PDL-related instructions:

```console
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM90=1
```

Note that this only adds PDL-related instructions to the _kernels_, but to actually allow a dependent
launch, you must also run your GEMM kernel with PDL:

```console
gemm.run(
  /* stream = */ stream,
  /* cuda_adapter = */ nullptr,
  /* launch_with_pdl = */ true
);_
```
