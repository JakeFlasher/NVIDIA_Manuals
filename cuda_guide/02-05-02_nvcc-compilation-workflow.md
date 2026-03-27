---
title: "2.5.2. NVCC Compilation Workflow"
section: "2.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-compilation-workflow"
---

## [2.5.2. NVCC Compilation Workflow](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#nvcc-compilation-workflow)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#nvcc-compilation-workflow "Permalink to this headline")

In the initial phase, `nvcc` separates the device code from the host code and dispatches their compilation to the GPU and the host compilers, respectively.

| To compile the host code, the CUDA compiler `nvcc` requires a compatible host compiler to be available. The CUDA Toolkit defines the host compiler support policy for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements) platforms.
| Files containing only host code can be built using either `nvcc` or the host compiler directly. The resulting object files can be combined with object files from `nvcc` which contain GPU code at link-time.

| The GPU compiler compiles the C/C++ device code to PTX assembly code. The GPU compiler is run for each virtual machine instruction set architecture (e.g. `compute_90`) specified in the compilation command line.
| Individual PTX code is then passed to the `ptxas` tool, which generates [Cubin](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-cubins-fatbins) for the target hardware ISAs. The hardware ISA is identified by its [SM version](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-compute-capability-sm-version).
| It is possible to embed multiple PTX and Cubin targets into a single binary [Fatbin](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-cubins-fatbins) container within an application or library so that a single binary can support multiple virtual and target hardware ISAs.

The invocation and coordination of the tools described above are done automatically by `nvcc`. The `-v` option can be used to display the full compilation workflow and tool invocation. The `-keep` option can be used to save the [intermediate files](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keeping-intermediate-phase-files) generated during the compilation in the current directory or in the directory specified by `--keep-dir` instead.

The following example illustrates the compilation workflow for a CUDA source file `example.cu`:

```cuda
// ----- example.cu -----
#include <stdio.h>
__global__ void kernel() {
    printf("Hello from kernel\n");
}

void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main() {
    kernel_launcher();
    return 0;
}
```

`nvcc` basic compilation workflow:

![High-level nvcc flow](images/____-___________-w______w_1.png)

`nvcc` compilation workflow with multiple PTX and Cubin architectures:

![High-level nvcc flow multiple architectures](images/____-___________-w______w_2.png)

A more detailed description of the `nvcc` compilation workflow can be found in the [compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory).
