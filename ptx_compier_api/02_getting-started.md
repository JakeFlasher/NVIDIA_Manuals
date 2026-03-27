---
title: "2. Getting Started"
section: "2"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#getting-started"
---

# [2. Getting Started](https://docs.nvidia.com/cuda/ptx-compiler-api#getting-started)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#getting-started "Permalink to this headline")

## [2.1. System Requirements](https://docs.nvidia.com/cuda/ptx-compiler-api#system-requirements)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#system-requirements "Permalink to this headline")

PTX Compiler library requires the following system configuration:

- POSIX threads support for non-Windows platform.
- GPU: Any GPU with CUDA Compute Capability 5.0 or higher.
- CUDA Toolkit and Driver.

## [2.2. Installation](https://docs.nvidia.com/cuda/ptx-compiler-api#installation)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#installation "Permalink to this headline")

PTX Compiler library is part of the CUDA Toolkit release and the components are organized as follows
in the CUDA toolkit installation directory:

- On Windows:
  - `include\nvPTXCompiler.h`
  - `lib\x64\nvptxcompiler_static.lib`
  - `doc\pdf\PTX_Compiler_API_User_Guide.pdf`
- On Linux:
  - `include/nvPTXCompiler.h`
  - `lib64/libnvptxcompiler_static.a`
  - `doc/pdf/PTX_Compiler_API_User_Guide.pdf`
