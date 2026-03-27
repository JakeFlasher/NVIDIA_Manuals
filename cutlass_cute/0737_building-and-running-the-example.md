---
title: "Building and Running the Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#building-and-running-the-example"
---

## [Building and Running the Example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#building-and-running-the-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#building-and-running-the-example "Permalink to this headline")

Example `09_turing_tensorop_conv2dfprop` computes a forward convolutional layer in which inputs and
outputs are 4-b integers. The example source is visible in
[examples/09_turing_tensorop_conv2dfprop/turing_tensorop_conv2dfprop.cu](https://github.com/NVIDIA/cutlass/tree/main/examples/09_turing_tensorop_conv2dfprop/turing_tensorop_conv2dfprop.cu).

Before building the example, first perform the prerequisite steps for building any CUTLASS component [described here](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html).
Compute capability 7.5 refers to the Turing architecture, and this work requires CUDA 10.2 Toolkit or later to target
Turing Tensor Cores using the native `mma` [PTX instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8832).

```bash
$ mkdir build && cd build

$ cmake .. -DCUTLASS_NVCC_ARCHS=75
```

To build the example, execute `make 09_turing_tensorop_conv2dfprop` from the build directory.

```bash
$ make 09_turing_tensorop_conv2dfprop

$ ls examples/09_turing_tensorop_conv2dfprop
examples/09_turing_tensorop_conv2dfprop
```

This example provides a simple command line interface to specify the extents of 4D tensors of 4-bit integer elements (`cutlass::int4b_t`),
initialize them to random values, and compute the result of a convolutional layer. Optionally, the input and output
tensors may be saved to .csv files, and the CUTLASS host-side reference check may be executed to verify correctness.

The complete usage statement is visible by running with `--help`:

```console
$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop --help
09_turing_tensorop_conv2dfprop example

  This example uses Turing's Tensor Core operators on int4 data types to compute
  forward convolution on tensors of layout NHWC.

Options:

  --help               If specified, displays this usage statement.

  --n <int>            Input tensor extent N
  --h <int>            Input tensor extent H
  --w <int>            Input tensor extent W
  --c <int>            Input tensor extent C
  --k <int>            Filter extent K
  --r <int>            Filter extent R
  --s <int>            Filter extent S

  --alpha <float>      Epilogue scalar alpha
  --beta <float>       Epilogue scalar beta

  --ref-check          If set (true), reference check on the host is computed
  --perf-check         If set (true), performance is measured.
  --benchmark          If set (true), performance benchmarking on several layers and batch-size.
  --iterations <int>   Number of profiling iterations to perform.
  --save-workspace     If set, workspace is written to a text file.
  --tag <string>       String to replicate across the first column in the results table

Examples:

$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1

$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check
```

_Note_, this example assumes all tensors are 128b aligned and in format _NHWC_. Consequently, dimension
_C_ must be divisible by 32 for activations, filters, and output.

If the option `--benchmark` is passed, several layers from ResNet50 are profiled for various batch sizes.
This sample output was computed on an NVIDIA RTX 2080 compiled with CUDA 10.2.

```bash
build$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop --benchmark
```

Convolution can also be run by the CUTLASS Profiler.
