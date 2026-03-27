---
title: "1. Introduction"
section: "1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction"
---

# [1. Introduction](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#introduction)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction "Permalink to this headline")

## [1.1. Overview](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#overview)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#overview "Permalink to this headline")

### [1.1.1. CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cuda-programming-model)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-programming-model "Permalink to this headline")

The CUDA Toolkit targets a class of applications whose control part runs as a process on a general purpose computing device, and which use one or more NVIDIA GPUs as coprocessors for accelerating _single program, multiple data_ (SPMD) parallel jobs. Such jobs are self-contained, in the sense that they can be executed and completed by a batch of GPU threads entirely without intervention by the host process, thereby gaining optimal benefit from the parallel graphics hardware.

The GPU code is implemented as a collection of functions in a language that is essentially C++, but with some annotations for distinguishing them from the host code, plus annotations for distinguishing different types of data memory that exists on the GPU. Such functions may have parameters, and they can be called using a syntax that is very similar to regular C function calling, but slightly extended for being able to specify the matrix of GPU threads that must execute the called function. During its life time, the host process may dispatch many parallel GPU tasks.

For more information on the CUDA programming model, consult the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

### [1.1.2. CUDA Sources](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cuda-sources)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-sources "Permalink to this headline")

Source files for CUDA applications consist of a mixture of conventional C++ host code, plus GPU device functions. The CUDA compilation trajectory separates the device functions from the host code, compiles the device functions using the proprietary NVIDIA compilers and assembler, compiles the host code using a C++ host compiler that is available, and afterwards embeds the compiled GPU functions as fatbinary images in the host object file. In the linking stage, specific CUDA runtime libraries are added for supporting remote SPMD procedure calling and for providing explicit GPU manipulation such as allocation of GPU memory buffers and host-GPU data transfer.

### [1.1.3. Purpose of NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#purpose-of-nvcc)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#purpose-of-nvcc "Permalink to this headline")

The compilation trajectory involves several splitting, compilation, preprocessing, and merging steps for each CUDA source file. It is the purpose of `nvcc`, the CUDA compiler driver, to hide the intricate details of CUDA compilation from developers. It accepts a range of conventional compiler options, such as for defining macros and include/library paths, and for steering the compilation process. All non-CUDA compilation steps are forwarded to a C++ host compiler that is supported by `nvcc`, and `nvcc` translates its options to appropriate host compiler command line options.

## [1.2. Supported Host Compilers](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#supported-host-compilers)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-host-compilers "Permalink to this headline")

A general purpose C++ host compiler is needed by `nvcc` in the following situations:

- During non-CUDA phases (except the run phase), because these phases will be forwarded by `nvcc` to this compiler.
- During CUDA phases, for several preprocessing stages and host code compilation (see also [The CUDA Compilation Trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-trajectory)).

`nvcc` assumes that the host compiler is installed with the standard method designed by the compiler provider. If the host compiler installation is non-standard, the user must make sure that the environment is set appropriately and use relevant `nvcc` compile options.

The following documents provide detailed information about supported host compilers:

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

On all platforms, the default host compiler executable (`gcc` and `g++` on Linux and `cl.exe` on Windows) found in the current execution search path will be used, unless specified otherwise with appropriate options (see *file-and-path-specifications*).

Note, `nvcc` does not support the compilation of file paths that exceed the maximum path length limitations of the host system. To support the compilation of long file paths, please refer to the documentation for your system.
