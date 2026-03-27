---
title: "5.5.1.8. Notes on Host/Device Computation Accuracy"
section: "5.5.1.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#notes-on-host-device-computation-accuracy"
---

### [5.5.1.8. Notes on Host/Device Computation Accuracy](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#notes-on-host-device-computation-accuracy)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#notes-on-host-device-computation-accuracy "Permalink to this headline")

The accuracy of a floating-point computation result is affected by several factors. This section summarizes important considerations for achieving reliable results in floating-point computations. Some of these aspects have been described in greater detail in previous sections.

These aspects are also important when comparing the results between CPU and GPU. Differences between host and device execution must be interpreted carefully. The presence of differences does not necessarily mean the GPU’s result is incorrect or that there is a problem with the GPU.

**Associativity**:

> Floating-point addition and multiplication in finite precision are not [associative](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#associativity) because they often result in mathematical values that cannot be directly represented in the target format, requiring rounding. The order in which these operations are evaluated affects how rounding errors accumulate and can significantly alter the final result.

**Fused Multiply-Add**:

> [Fused Multiply-Add](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#fused-multiply-add) computes \(\(a \times b + c\)\) in a single operation, resulting in greater accuracy and a faster execution time. The accuracy of the final result can be affected by its use. Fused Multiply-Add relies on hardware support and can be enabled either explicitly by calling the related function or implicitly through compiler optimization flags.

**Precision**:

> Increasing the floating-point precision can potentially improve the accuracy of the results. Higher precision reduces loss of significance and enables the representation of a wider range of values. However, higher precision types have lower throughput and consume more registers. Additionally, using them to explicitly store input and output increases memory usage and data movement.

**Compiler Flags and Optimizations**:

> All major compilers provide a variety of optimization flags to control the behavior of floating-point operations.
>
>
> - The highest optimization level for GCC (`-O3`), Clang (`-O3`), nvcc (`-O3`), and Microsoft Visual Studio (`/O2`) does not affect floating-point semantics. However, inlining, loop unrolling, vectorization, and common subexpression elimination could affect the results. The NVC++ compiler also requires the flags `-Kieee -Mnofma` for IEEE-754-compliant semantics.
> - Refer to the [GCC](https://gcc.gnu.org/wiki/FloatingPointMath), [Clang](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior), [Microsoft Visual Studio Compiler](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior), [nvc++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and [Arm C/C++ compiler](https://developer.arm.com/documentation/101458/2404/Compiler-options?lang=en) documentation for detailed information about options that affect floating-point behavior.
> - See also the `nvcc` [User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) for detailed descriptions of compiler flags that specifically affect floating-point behavior in CUDA device code: `-ftz`, `-prec-div`, `-prec-sqrt`, `-fmad`, `--use_fast_math`. Besides these floating-point options, it is also important to verify the effects of other compiler optimizations in the context of the user program. Users are encouraged to verify the correctness of their results with extensive testing and compare results obtained with optimizations enabled versus all device code optimizations disabled; see also the `-G` compiler flag.

**Library Implementations**:

> Functions defined outside the IEEE-754 standard are not guaranteed to be correctly rounded and depend on implementation-defined behavior. Therefore, the results may differ across different platforms, including between host, device, and different device architectures.

**Deterministic Results**:

> A deterministic result refers to computing the same bit-wise numerical outputs every time when run with the same inputs under the same specified conditions. Such conditions include:
>
>
> - Hardware dependencies, such as execution on the same CPU processor or GPU device.
> - Compiler aspects, such as the version of the compiler and the [Compiler Flags and Optimizations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compiler-flags-and-optimizations).
> - Run-time conditions that affect the computation, such as [rounding mode](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-rounding) or environment variables.
> - Identical inputs to the computation.
> - Thread configuration, including the number of threads involved in the computation and their organization, for example block and grid size.
> - The ordering of [arithmetic atomic operations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions) depends on hardware scheduling which can vary between runs.

**Taking Advantage of the CUDA Libraries**:

> The [CUDA Math Libraries](https://developer.nvidia.com/gpu-accelerated-libraries), [C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html), and [C++ Standard Library Mathematical functions](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) are designed to boost developer productivity for common functionalities, particularly for floating-point math and numerics-intensive routines. These functionalities provide a consistent high-level interface, are optimized, and are widely tested across platforms and edge cases. Users are encouraged to take full advantage of these libraries and avoid tedious manual reimplementations.
