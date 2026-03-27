---
title: "2.5.4.3. Optimization Options"
section: "2.5.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#optimization-options"
---

### [2.5.4.3. Optimization Options](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#optimization-options)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#optimization-options "Permalink to this headline")

`nvcc` provides many options for optimizing performance. This section aims to provide a brief survey of some of the options available that developers may find useful, as well as links to further information. Complete coverage can be found in the [nvcc documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

- `-Xptxas` passes arguments to the PTX assembler tool `ptxas`. The `nvcc` documentation provides a [list of useful arguments](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options) for `ptxas`. For example, `-Xptxas=-maxrregcount=N` specifies the maximum number of registers to use, per thread.
- `-extra-device-vectorization`: Enables more aggressive device code vectorization.
- Additional flags which provide fine-grained control over floating point behavior are covered in the [Floating-Point Computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-computation) section and in the [nvcc documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).

The following flags get output from the compiler which can be useful in more advanced code optimization:

- `-res-usage`: Print resource usage report after compilation. It includes the number of registers, shared memory, constant memory, and local memory allocated for each kernel function.
- `-opt-info=inline`: Print information about inlined functions.
- `-Xptxas=-warn-lmem-usage`: Warn if local memory is used.
- `-Xptxas=-warn-spills`: Warn if registers are spilled to local memory.
