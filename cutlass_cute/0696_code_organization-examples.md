---
title: "Examples"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#code_organization--examples"
---

## [Examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#examples)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#examples "Permalink to this headline")

To demonstrate CUTLASS components, several SDK examples are implemented in `examples/`.

CUTLASS SDK examples apply CUTLASS templates to implement basic computations.

```console
examples/
  00_basic_gemm/             # launches a basic GEMM with single precision inputs and outputs

  01_cutlass_utilities/      # demonstrates CUTLASS Utilities for allocating and initializing tensors

  02_dump_reg_smem/          # debugging utilities for printing register and shared memory contents

  03_visualize_layout/       # utility for visualizing all layout functions in CUTLASS

  04_tile_iterator/          # example demonstrating an iterator over tiles in memory

  05_batched_gemm/           # example demonstrating CUTLASS's batched strided GEMM operation

  06_splitK_gemm/            # exmaple demonstrating CUTLASS's Split-K parallel reduction kernel

  07_volta_tensorop_gemm/    # example demonstrating mixed precision GEMM using Volta Tensor Cores

  08_turing_tensorop_gemm/   # example demonstrating integer GEMM using Turing Tensor Cores

  10_planar_complex/         # example demonstrating planar complex GEMM kernels

  11_planar_complex_array/   # example demonstrating planar complex kernels with batch-specific problem sizes

  12_gemm_bias_relu/         # example demonstrating GEMM fused with bias and relu activation function

  13_fused_two_gemms/        # example demonstrating two GEMMs fused into one kernel
```
