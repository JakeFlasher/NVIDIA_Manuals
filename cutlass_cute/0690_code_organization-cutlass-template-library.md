---
title: "CUTLASS Template Library"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#code_organization--cutlass-template-library"
---

## [CUTLASS Template Library](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-template-library)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-template-library "Permalink to this headline")

CUDA Templates for Linear Algebra Subroutines and Solvers is a library of CUDA C++ template classes for
performing efficient matrix computations on NVIDIA GPUs.

Like NVIDIA CUB, the components of CUTLASS are organized hierarchically based on the scope of cooperative
elements. For example, warp-level GEMM components perform a matrix multiply collectively by the
set of threads within a warp. The following figure illustrates each layer.

Components are designed to be usable by client applications accessing functionailty at each scope.

CUTLASS Templates are implemented by header files in the following directory structure:

```console
include/                     # Top-level include directory. Client applications should target this path.
  cutlass/                   # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)
      *
    gemm/                    # code specialized for general matrix product computations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      collective/            #   3.x API operators for all threads a tiled mma/copy are built over
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions for GEMM

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory
      *

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" models
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions

    transform/               # code specialized for layout, type, and domain transformations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions

    util/                    # miscellaneous CUTLASS components
      *
    *                        # core vocabulary types and fundamental arithmetic operators

  cute /                     # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy
    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples
    arch/                    # Bare bones PTX wrapper structs for copy and math instructions
    atom/                    # Meta-information either link to or built from arch/ operators
      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma
      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy
      *sm*.hpp               # Arch specific meta-information for copy and math operations
    container/               # Core container types used across CuTe, namely, cute::tuple
    numeric/                 # CuTe's internal numerics implementation
    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations
```

See [Programming Guidelines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html) for further details about
conventions and design patterns used throughout CUTLASS.
