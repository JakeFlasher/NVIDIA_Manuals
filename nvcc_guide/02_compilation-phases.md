---
title: "2. Compilation Phases"
section: "2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases"
---

# [2. Compilation Phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compilation-phases)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases "Permalink to this headline")

## [2.1. NVCC Identification Macro](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-identification-macro)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-identification-macro "Permalink to this headline")

`nvcc` predefines the following macros:

**`__NVCC__`**

  Defined when compiling C/C++/CUDA source files.

**`__CUDACC__`**

  Defined when compiling CUDA source files.

**`__CUDACC_RDC__`**

  Defined when compiling CUDA source files in relocatable device code mode (see [NVCC Options for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-options-for-separate-compilation)).

**`__CUDACC_EWP__`**

  Defined when compiling CUDA source files in extensible whole program mode (see [Options for Specifying Behavior of Compiler/Linker](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-altering-compiler-linker-behavior)).

**`__CUDACC_DEBUG__`**

  Defined when compiling CUDA source files in the device-debug mode (see [Options for Specifying Behavior of Compiler/Linker](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-altering-compiler-linker-behavior)).

**`__CUDACC_RELAXED_CONSTEXPR__`**

  Defined when the `--expt-relaxed-constexpr` flag is specified on the command line. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.

**`__CUDACC_EXTENDED_LAMBDA__`**

  Defined when the `--expt-extended-lambda` or `--extended-lambda` flag is specified on the command line. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.

**`__CUDACC_VER_MAJOR__`**

  Defined with the major version number of `nvcc`.

**`__CUDACC_VER_MINOR__`**

  Defined with the minor version number of `nvcc`.

**`__CUDACC_VER_BUILD__`**

  Defined with the build version number of `nvcc`.

**`__NVCC_DIAG_PRAGMA_SUPPORT__`**

  Defined when the CUDA frontend compiler supports diagnostic control with the `nv_diag_suppress`, `nv_diag_error`, `nv_diag_warning`, `nv_diag_default`, `nv_diag_once`, and `nv_diagnostic` pragmas.

**`__CUDACC_DEVICE_ATOMIC_BUILTINS__`**

  Defined when the CUDA frontend compiler supports device atomic compiler builtins. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.

## [2.2. NVCC Phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-phases)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-phases "Permalink to this headline")

A compilation phase is a logical translation step that can be selected by command line options to `nvcc`. A single compilation phase can still be broken up by `nvcc` into smaller steps, but these smaller steps are just implementations of the phase: they depend on seemingly arbitrary capabilities of the internal tools that `nvcc` uses, and all of these internals may change with a new release of the CUDA Toolkit. Hence, only compilation phases are stable across releases, and although `nvcc` provides options to display the compilation steps that it executes, these are for debugging purposes only and must not be copied and used in build scripts.

`nvcc` phases are selected by a combination of command line options and input file name suffixes, and the execution of these phases may be modified by other command line options. In phase selection, the input file suffix defines the phase input, while the command line option defines the required output of the phase.

The following paragraphs list the recognized file name suffixes and the supported compilation phases. A full explanation of the `nvcc` command line options can be found in [NVCC Command Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-command-options).

## [2.3. Supported Input File Suffixes](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#supported-input-file-suffixes)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-input-file-suffixes "Permalink to this headline")

The following table defines how `nvcc` interprets its input files:

| Input File Suffix | Description |
| --- | --- |
| `.cu` | CUDA source file, containing host code and device functions |
| `.c` | C source file |
| `.cc`, `.cxx`, `.cpp` | C++ source file |
| `.ptx` | PTX intermediate assembly file (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.cubin` | CUDA device code binary file (CUBIN) for a single GPU architecture (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.fatbin` | CUDA fat binary file that may contain multiple PTX and CUBIN files (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.o`, `.obj` | Object file |
| `.a`, `.lib` | Library file |
| `.res` | Resource file |
| `.so` | Shared object file |

Note that `nvcc` does not make any distinction between object, library or resource files. It just passes files of these types to the linker when the linking phase is executed.

## [2.4. Supported Phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#supported-phases)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-phases "Permalink to this headline")

The following table specifies the supported compilation phases, plus the option to `nvcc` that enables the execution of each phase. It also lists the default name of the output file generated by each phase, which takes effect when no explicit output file name is specified using the option `--output-file`:

| Phase | `nvcc` Option |  | Default Output File Name |
| --- | --- | --- | --- |
|  | Long Name | Short Name |  |
| CUDA compilation to C/C++ source file | `--cuda` | `-cuda` | `.cpp.ii` appended to source file name, as in `x.cu.cpp.ii`. This output file can be compiled by the host compiler that was used by `nvcc` to preprocess the `.cu` file. |
| C/C++ preprocessing | `--preprocess` | `-E` | <_result on standard output_> |
| C/C++ compilation to object file | `--compile` | `-c` | Source file name with suffix replaced by `o` on Linux or `obj` on Windows |
| Cubin generation from CUDA source files | `--cubin` | `-cubin` | Source file name with suffix replaced by `cubin` |
| Cubin generation from PTX intermediate files. | `--cubin` | `-cubin` | Source file name with suffix replaced by `cubin` |
| PTX generation from CUDA source files | `--ptx` | `-ptx` | Source file name with suffix replaced by `ptx` |
| Fatbinary generation from source, PTX or cubin files | `--fatbin` | `-fatbin` | Source file name with suffix replaced by `fatbin` |
| Linking relocatable device code. | `--device-link` | `-dlink` | `a_dlink.obj` on Windows or `a_dlink.o` on other platforms |
| Cubin generation from linked relocatable device code. | `--device-link``--cubin` | `-dlink``-cubin` | `a_dlink.cubin` |
| Fatbinary generation from linked relocatable device code | `--device-link``--fatbin` | `-dlink``-fatbin` | `a_dlink.fatbin` |
| Linking an executable | <_no phase option_> |  | `a.exe` on Windows or `a.out` on other platforms |
| Constructing an object file archive, or library | `--lib` | `-lib` | `a.lib` on Windows or `a.a` on other platforms |
| `make` dependency generation | `--generate-dependencies` | `-M` | <_result on standard output_> |
| `make` dependency generation without headers in system paths. | `--generate-nonsystem-dependencies` | `-MM` | <_result on standard output_> |
| Compile CUDA source to OptiX IR output. | `--optix-ir` | `-optix-ir` | Source file name with suffix replaced by `optixir` |
| Compile CUDA source to LTO IR output. | `--ltoir` | `-ltoir` | Source file name with suffix replaced by `ltoir` |
| Running an executable | `--run` | `-run` |  |

**Notes:**

- The last phase in this list is more of a convenience phase. It allows running the compiled and linked executable without having to explicitly set the library path to the CUDA dynamic libraries.
- Unless a phase option is specified, `nvcc` will compile and link all its input files.
