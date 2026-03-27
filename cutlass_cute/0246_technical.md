---
title: "Technical"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html#technical"
---

## [Technical](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#technical)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#technical "Permalink to this headline")

**What NVIDIA architectures will it support?**

> CuTe DSL will support all NVIDIA GPU architectures starting with NVIDIA Ampere Architecture (SM80).

**Will it be compatible with DL frameworks (e.g., PyTorch, JAX)?**

> Yes, we will provide utilities to convert from DLPack-supported tensor formats
> to `cute.Tensor`. This should allow a user to never have to leave Python
> when writing model code in their framework of choice. Our JAX interoperability story is not
> as strong as PyTorch’s today, however, we are actively working on improving it
> and welcome contributions in this space.

**Does it compile to PTX or SASS?**

> CuTe DSL compiles the program down to PTX. After that, we currently use the PTX compiler that
> ships with the CUDA toolkit to compile the PTX down to SASS. We plan to remove
> this limitation in the future and allow the use of the PTX JIT that is included in the
> CUDA driver in case a user does not have a CUDA toolkit installed.

**Do I need to use NVCC or NVRTC?**

> No, the `nvidia-cutlass-dsl` wheel packages is everything needed to generate GPU kernels. It
> shares the driver requirements of the 12.9 toolkit which can be found
> [here](https://developer.nvidia.com/cuda-toolkit-archive).

**How would one debug the code?**

> Since CuTe DSL is not native python and an embedded DSL instead, tools like *pdb*
> cannot be used.  However, if you have experience with GPU kernel programming, the debugging
> techniques will be nearly identical. Typically, compile time and runtime printing
> of types and values are the most expedient. Please see [documentation on printing](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb)
> to learn how to print types and values at both compile time and runtime.
> You can also use `cuda-gdb` to set breakpoints in the program and step through the execution
> or use tools such as `compute-sanitizer` to detect and triage bugs in your program. As the DSL
> matures, our source location tracking from Python user programs will also improve to provide
> more helpful source-level mapping when setting breakpoints and using other tools such as nsight.

**How would one implement warp specialization in CuTe DSL?**

> Exactly the same way you would in C++ but in a Python-native syntax instead.
> Consult our [Control Flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html) and
> [“Blackwell kernel example”](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)
> for a detailed how-to guide.

**Can I call functions from other functions or use OOP?**

> Yes. We frequently call functions from one another and set up class
> hierarchies to organize and modularize our code for pipelines and schedulers.
> Consult the [Introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) documentation or our examples for more details.
