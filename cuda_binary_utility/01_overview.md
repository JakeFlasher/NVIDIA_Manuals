---
title: "1. Overview"
section: "1"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#overview"
---

# [1. Overview](https://docs.nvidia.com/cuda/cuda-binary-utilities#overview)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#overview "Permalink to this headline")

This document introduces `cuobjdump`, `nvdisasm`, `cu++filt` and `nvprune`, four CUDA binary tools for Linux (x86, ARM and P9), Windows, Mac OS and Android.

## [1.1. What is a CUDA Binary?](https://docs.nvidia.com/cuda/cuda-binary-utilities#what-is-a-cuda-binary)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#what-is-a-cuda-binary "Permalink to this headline")

A CUDA binary (also referred to as cubin) file is an ELF-formatted file which consists of CUDA executable code sections as well as other sections containing symbols, relocators, debug info, etc. By default, the CUDA compiler driver `nvcc` embeds cubin files into the host executable file. But they can also be generated separately by using the “`-cubin`” option of `nvcc`. cubin files are loaded at run time by the CUDA driver API.

> **Note**
>
> For more details on cubin files or the CUDA compilation trajectory, refer to [NVIDIA CUDA Compiler Driver NVCC](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## [1.2. Differences between cuobjdump and nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities#differences-between-cuobjdump-and-nvdisasm)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#differences-between-cuobjdump-and-nvdisasm "Permalink to this headline")

CUDA provides two binary utilities for examining and disassembling cubin files and host executables: `cuobjdump` and `nvdisasm`. Basically, `cuobjdump` accepts both cubin files and host binaries while `nvdisasm` only accepts cubin files; but `nvdisasm` provides richer output options.

Here’s a quick comparison of the two tools:

|  | `cuobjdump` | `nvdisasm` |
| --- | --- | --- |
| Disassemble cubin | Yes | Yes |
| Extract ptx and extract and disassemble cubin from the following input files:   - Host binaries   - Executables   - Object files   - Static libraries - External fatbinary files | Yes | No |
| Control flow analysis and output | No | Yes |
| Advanced display options | No | Yes |

## [1.3. Command Option Types and Notation](https://docs.nvidia.com/cuda/cuda-binary-utilities#command-option-types-and-notation)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#command-option-types-and-notation "Permalink to this headline")

This section of the document provides common details about the command line options for the following tools:

- [cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump)
- [nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvdisasm)
- [nvprune](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune)

Each command-line option has a long name and a short name, which are interchangeable with each other. These two variants are distinguished by the number of hyphens that must precede the option name, i.e. long names must be preceded by two hyphens and short names must be preceded by a single hyphen. For example, `-I` is the short name of `--include-path`. Long options are intended for use in build scripts, where size of the option is less important than descriptive value and short options are intended for interactive use.

The tools mentioned above recognize three types of command options: boolean options, single value options and list options.

Boolean options do not have an argument, they are either specified on a command line or not. Single value options must be specified at most once and list options may be repeated. Examples of each of these option types are, respectively:

```text
Boolean option : nvdisasm --print-raw <file>
Single value   : nvdisasm --binary SM100 <file>
List options   : cuobjdump --function "foo,bar,foobar" <file>
```

Single value options and list options must have arguments, which must follow the name of the option by either one or more spaces or an equals character. When a one-character short name such as `-I`, `-l`, and `-L` is used, the value of the option may also immediately follow the option itself without being seperated by spaces or an equal character. The individual values of list options may be separated by commas in a single instance of the option or the option may be repeated, or any combination of these two cases.

Hence, for the two sample options mentioned above that may take values, the following notations are legal:

```text
-o file
-o=file
-Idir1,dir2 -I=dir3 -I dir4,dir5
```

For options taking a single value, if specified multiple times, the rightmost value in the command line will be considered for that option. In the below example, `test.bin` binary will be disassembled assuming `SM120` as the architecture.

```text
nvdisasm.exe -b SM100 -b SM120 test.bin
nvdisasm warning : incompatible redefinition for option 'binary', the last value of this option was used
```

For options taking a list of values, if specified multiple times, the values get appended to the list. If there are duplicate values specified, they are ignored. In the below example, functions `foo` and `bar` are considered as valid values for option `--function` and the duplicate value `foo` is ignored.

```text
cuobjdump --function "foo" --function "bar" --function "foo" -sass  test.cubin
```
