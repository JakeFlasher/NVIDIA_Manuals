---
title: "4. NVCC Command Options"
section: "4"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-command-options"
---

# [4. NVCC Command Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-command-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-command-options "Permalink to this headline")

## [4.1. Command Option Types and Notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#command-option-types-and-notation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#command-option-types-and-notation "Permalink to this headline")

Each `nvcc` option has a long name and a short name, which are interchangeable with each other. These two variants are distinguished by the number of hyphens that must precede the option name: long names must be preceded by two hyphens, while short names must be preceded by a single hyphen. For example, `-I` is the short name of `--include-path`. Long options are intended for use in build scripts, where the size of the option is less important than the descriptive value. In contrast, short options are intended for interactive use.

`nvcc` recognizes three types of command options: boolean options, single value options, and list options.

Boolean options do not have an argument; they are either specified on the command line or not. Single value options must be specified at most once, but list options may be repeated. Examples of each of these option types are, respectively: `--verbose` (switch to verbose mode), `--output-file` (specify output file), and `--include-path` (specify include path).

Single value options and list options must have arguments, which must follow the name of the option itself by either one of more spaces or an equals character. When a one-character short name such as `-I`, `-l`, and `-L` is used, the value of the option may also immediately follow the option itself without being seperated by spaces or an equal character. The individual values of list options may be separated by commas in a single instance of the option, or the option may be repeated, or any combination of these two cases.

Hence, for the two sample options mentioned above that may take values, the following notations are legal:

```text
-o file
```

```text
-o=file
```

```text
-Idir1,dir2 -I=dir3 -I dir4,dir5
```

Unless otherwise specified, long option names are used throughout this document. However, short names can be used instead of long names for the same effect.

## [4.2. Command Option Description](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#command-option-description)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#command-option-description "Permalink to this headline")

This section presents tables of `nvcc` options. The option type in the tables can be recognized as follows: Boolean options do not have arguments specified in the first column, while the other two types do. List options can be recognized by the repeat indicator `,...` at the end of the argument.

Long options are described in the first column of the options table, and short options occupy the second column.

### [4.2.1. File and Path Specifications](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#file-and-path-specifications)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#file-and-path-specifications "Permalink to this headline")

#### [4.2.1.1. --output-file file (-o)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#output-file-file-o)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#output-file-file-o "Permalink to this headline")

Specify name and location of the output file.

#### [4.2.1.2. --objdir-as-tempdir (-objtemp)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#objdir-as-tempdir-objtemp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#objdir-as-tempdir-objtemp "Permalink to this headline")

Create all intermediate files in the same directory as the object file. These intermediate files are deleted when the compilation is finished. This option will take effect only if -c, -dc or -dw is also used. Using this option will ensure that the intermediate file name that is embedded in the object file will not change in multiple compiles of the same file. However, this is not guaranteed if the input is stdin. If the same file is compiled with two different options, ex., ‘nvcc -c t.cu’ and ‘nvcc -c -ptx t.cu’, then the files should be compiled in different directories. Compiling them in the same directory can either cause the compilation to fail or produce incorrect results.

#### [4.2.1.3. --pre-include file,... (-include)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#pre-include-file-include)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#pre-include-file-include "Permalink to this headline")

Specify header files that must be pre-included during preprocessing.

#### [4.2.1.4. --library library,... (-l)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#library-library-l)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#library-library-l "Permalink to this headline")

Specify libraries to be used in the linking stage without the library file extension.

The libraries are searched for on the library search paths that have been specified using option `--library-path` (see [Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libraries)).

#### [4.2.1.5. --define-macro def,... (-D)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#define-macro-def-d)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#define-macro-def-d "Permalink to this headline")

Define macros to be used during preprocessing.

_def_ can be either _name_ or _name_=_definition_.

- _name_
-  Predefine _name_ as a macro.
- _name_=_definition_
-  The contents of _definition_ are tokenized and preprocessed as if they appear during translation phase three in a `#define` directive. The definition will be truncated by embedded new line characters.

#### [4.2.1.6. --undefine-macro def,... (-U)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#undefine-macro-def-u)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#undefine-macro-def-u "Permalink to this headline")

Undefine an existing macro during preprocessing or compilation.

#### [4.2.1.7. --include-path path,... (-I)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#include-path-path-i)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#include-path-path-i "Permalink to this headline")

Specify include search paths.

#### [4.2.1.8. --system-include path,... (-isystem)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#system-include-path-isystem)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#system-include-path-isystem "Permalink to this headline")

Specify system include search paths.

#### [4.2.1.9. --library-path path,... (-L)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#library-path-path-l)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#library-path-path-l "Permalink to this headline")

Specify library search paths (see [Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libraries)).

#### [4.2.1.10. --output-directory directory (-odir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#output-directory-directory-odir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#output-directory-directory-odir "Permalink to this headline")

Specify the directory of the output file.

This option is intended for letting the dependency generation step (see `--generate-dependencies`) generate a rule that defines the target object file in the proper directory.

#### [4.2.1.11. --dependency-output file (-MF)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dependency-output-file-mf)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dependency-output-file-mf "Permalink to this headline")

Specify the dependency output file.

This option specifies the output file for the dependency generation step (see `--generate-dependencies`). The option `--generate-dependencies` or `--generate-nonystem-dependencies` must be specified if a dependency output file is set.

#### [4.2.1.12. --generate-dependency-targets (-MP)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-dependency-targets-mp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-dependency-targets-mp "Permalink to this headline")

Add an empty target for each dependency.

This option adds phony targets to the dependency generation step (see `--generate-dependencies`) intended to avoid makefile errors if old dependencies are deleted. The input files are not emitted as phony targets.

#### [4.2.1.13. --compiler-bindir directory (-ccbin)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compiler-bindir-directory-ccbin)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compiler-bindir-directory-ccbin "Permalink to this headline")

Specify the directory in which the default host compiler executable resides.

The host compiler executable name can be also specified to ensure that the correct host compiler is selected. In addition, driver prefix options (`--input-drive-prefix`, `--dependency-drive-prefix`, or `--drive-prefix`) may need to be specified, if `nvcc` is executed in a Cygwin shell or a MinGW shell on Windows.

#### [4.2.1.14. --allow-unsupported-compiler (-allow-unsupported-compiler)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#allow-unsupported-compiler-allow-unsupported-compiler)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#allow-unsupported-compiler-allow-unsupported-compiler "Permalink to this headline")

Disable nvcc check for supported host compiler versions.

Using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk. This option has no effect on MacOS.

#### [4.2.1.15. --archiver-binary executable (-arbin)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#archiver-binary-executable-arbin)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#archiver-binary-executable-arbin "Permalink to this headline")

Specify the path of the archiver tool used create static library with `--lib`.

#### [4.2.1.16. --cudart {none|shared|static |hybrid} (-cudart)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cudart-none-shared-static-hybrid-cudart)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cudart-none-shared-static-hybrid-cudart "Permalink to this headline")

Specify the type of CUDA runtime library to be used: no CUDA runtime library, shared/dynamic CUDA runtime library, or static CUDA runtime library.
On Windows, the shared option has been replaced by a hybrid option, where a small loader library is statically linked in that dynamically loads the runtime from the Display Driver.

**Allowed Values**

- `none`
- `shared` (non-Windows)
- `static`
- `hybrid` (Windows)

**Default**

The static CUDA runtime library is used by default except on Windows, where the hybrid approach is the default instead.

#### [4.2.1.17. --cudadevrt {none|static} (-cudadevrt)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cudadevrt-none-static-cudadevrt)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cudadevrt-none-static-cudadevrt "Permalink to this headline")

Specify the type of CUDA device runtime library to be used: no CUDA device runtime library, or static CUDA device runtime library.

**Allowed Values**

- `none`
- `static`

**Default**

The static CUDA device runtime library is used by default.

#### [4.2.1.18. --libdevice-directory directory (-ldir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#libdevice-directory-directory-ldir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libdevice-directory-directory-ldir "Permalink to this headline")

Specify the directory that contains the libdevice library files.

Libdevice library files are located in the `nvvm/libdevice` directory in the CUDA Toolkit.

#### [4.2.1.19. --target-directory string (-target-dir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#target-directory-string-target-dir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#target-directory-string-target-dir "Permalink to this headline")

Specify the subfolder name in the targets directory where the default include and library paths are located.

### [4.2.2. Options for Specifying the Compilation Phase](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-specifying-the-compilation-phase)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-specifying-the-compilation-phase "Permalink to this headline")

Options of this category specify up to which stage the input files must be compiled.

#### [4.2.2.1. --link (-link)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#link-link)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#link-link "Permalink to this headline")

Specify the default behavior: compile and link all input files.

**Default Output File Name**

`a.exe` on Windows or `a.out` on other platforms is used as the default output file name.

#### [4.2.2.2. --lib (-lib)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#lib-lib)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#lib-lib "Permalink to this headline")

Compile all input files into object files, if necessary, and add the results to the specified library output file.

**Default Output File Name**

`a.lib` on Windows or `a.a` on other platforms is used as the default output file name.

#### [4.2.2.3. --device-link (-dlink)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-link-dlink)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-link-dlink "Permalink to this headline")

Link object files with relocatable device code and `.ptx`, `.cubin`, and `.fatbin` files into an object file with executable device code, which can be passed to the host linker.

**Default Output File Name**

`a_dlink.obj` on Windows or `a_dlink.o` on other platforms is used as the default output file name. When this option is used in conjunction with `--fatbin`, `a_dlink.fatbin` is used as the default output file name. When this option is used in conjunction with `--cubin`, `a_dlink.cubin` is used as the default output file name.

#### [4.2.2.4. --device-c (-dc)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-c-dc)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-c-dc "Permalink to this headline")

Compile each `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input file into an object file that contains relocatable device code.

It is equivalent to `--relocatable-device-code=true --compile`.

**Default Output File Name**

The source file name extension is replaced by `.obj` on Windows and `.o` on other platforms to create the default output file name. For example, the default output file name for `x.cu` is `x.obj` on Windows and `x.o` on other platforms.

#### [4.2.2.5. --device-w (-dw)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-w-dw)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-w-dw "Permalink to this headline")

Compile each `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input file into an object file that contains executable device code.

It is equivalent to `--relocatable-device-code=false --compile`.

**Default Output File Name**

The source file name extension is replaced by `.obj` on Windows and `.o` on other platforms to create the default output file name. For example, the default output file name for `x.cu` is `x.obj` on Windows and `x.o` on other platforms.

#### [4.2.2.6. --cuda (-cuda)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cuda-cuda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-cuda "Permalink to this headline")

Compile each `.cu` input file to a `.cu.cpp.ii` file.

**Default Output File Name**

`.cu.cpp.ii` is appended to the basename of the source file name to create the default output file name. For example, the default output file name for `x.cu` is `x.cu.cpp.ii`.

#### [4.2.2.7. --compile (-c)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compile-c)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compile-c "Permalink to this headline")

Compile each `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input file into an object file.

**Default Output File Name**

The source file name extension is replaced by `.obj` on Windows and `.o` on other platforms to create the default output file name. For example, the default output file name for `x.cu` is `x.obj` on Windows and `x.o` on other platforms.

#### [4.2.2.8. --fatbin (-fatbin)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fatbin-fatbin)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fatbin-fatbin "Permalink to this headline")

Compile all `.cu`, `.ptx`, and `.cubin` input files to device-only `.fatbin` files.

`nvcc` discards the host code for each `.cu` input file with this option.

**Default Output File Name**

The source file name extension is replaced by `.fatbin` to create the default output file name. For example, the default output file name for `x.cu` is `x.fatbin`.

#### [4.2.2.9. --cubin (-cubin)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cubin-cubin)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cubin-cubin "Permalink to this headline")

Compile all `.cu` and `.ptx` input files to device-only `.cubin` files.

`nvcc` discards the host code for each `.cu` input file with this option.

**Default Output File Name**

The source file name extension is replaced by `.cubin` to create the default output file name. For example, the default output file name for `x.cu` is `x.cubin`.

#### [4.2.2.10. --ptx (-ptx)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptx-ptx)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptx-ptx "Permalink to this headline")

Compile all `.cu` input files to device-only `.ptx` files.

`nvcc` discards the host code for each `.cu` input file with this option.

**Default Output File Name**

The source file name extension is replaced by `.ptx` to create the default output file name. For example, the default output file name for `x.cu` is `x.ptx`.

#### [4.2.2.11. --preprocess (-E)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#preprocess-e)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#preprocess-e "Permalink to this headline")

Preprocess all `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input files.

**Default Output File Name**

The output is generated in _stdout_ by default.

#### [4.2.2.12. --generate-dependencies (-M)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-dependencies-m)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-dependencies-m "Permalink to this headline")

Generate a dependency file that can be included in a `Makefile` for the `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input file.

`nvcc` uses a fixed prefix to identify dependencies in the preprocessed file ( ‘`#line 1`’ on Linux and ‘`# 1`’ on Windows). The files mentioned in source location directives starting with this prefix will be included in the dependency list.

**Default Output File Name**

The output is generated in _stdout_ by default.

#### [4.2.2.13. --generate-nonsystem-dependencies (-MM)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-nonsystem-dependencies-mm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-nonsystem-dependencies-mm "Permalink to this headline")

Same as `--generate-dependencies` but skip header files found in system directories (Linux only).

**Default Output File Name**

The output is generated in _stdout_ by default.

#### [4.2.2.14. --generate-dependencies-with-compile (-MD)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-dependencies-with-compile-md)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-dependencies-with-compile-md "Permalink to this headline")

Generate a dependency file and compile the input file. The dependency file can be included in a `Makefile` for the `.c`, `.cc`, `.cpp`, `.cxx`, and `.cu` input file.

This option cannot be specified together with `-E`. The dependency file name is computed as follows:

- If `-MF` is specified, then the specified file is used as the dependency file name.
- If `-o` is specified, the dependency file name is computed from the specified file name by replacing the suffix with ‘.d’.
- Otherwise, the dependency file name is computed by replacing the input file names’s suffix with ‘.d’.

If the dependency file name is computed based on either `-MF` or `-o`, then multiple input files are not supported.

#### [4.2.2.15. --generate-nonsystem-dependencies-with-compile (-MMD)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-nonsystem-dependencies-with-compile-mmd)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-nonsystem-dependencies-with-compile-mmd "Permalink to this headline")

Same as `--generate-dependencies-with-compile` but skip header files found in system directories (Linux only).

#### [4.2.2.16. --optix-ir (-optix-ir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#optix-ir-optix-ir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optix-ir-optix-ir "Permalink to this headline")

Compile CUDA source to OptiX IR (.optixir) output. The OptiX IR is only intended for consumption by OptiX through appropriate APIs. This feature is not supported with link-time-optimization (`-dlto`), the lto_NN -arch target, or with `-gencode`.

**Default Output File Name**

The source file name extension is replaced by `.optixir` to create the default output file name. For example, the default output file name for `x.cu` is `x.optixir`.

#### [4.2.2.17. --ltoir (-ltoir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ltoir-ltoir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ltoir-ltoir "Permalink to this headline")

Compile CUDA source to LTO IR (.ltoir) output. This feature is only supported with link-time-optimization (`-dlto`) or the lto_NN -arch target.

**Default Output File Name**

The source file name extension is replaced by `.ltoir` to create the default output file name. For example, the default output file name for `x.cu` is `x.ltoir`.

#### [4.2.2.18. --run (-run)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#run-run)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#run-run "Permalink to this headline")

Compile and link all input files into an executable, and executes it.

When the input is a single executable, it is executed without any compilation or linking. This step is intended for developers who do not want to be bothered with setting the necessary environment variables; these are set temporarily by `nvcc`.

### [4.2.3. Options for Specifying Behavior of Compiler/Linker](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-specifying-behavior-of-compiler-linker)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-specifying-behavior-of-compiler-linker "Permalink to this headline")

#### [4.2.3.1. --profile (-pg)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#profile-pg)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#profile-pg "Permalink to this headline")

Instrument generated code/executable for use by `gprof`.

#### [4.2.3.2. --debug (-g)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#debug-g)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#debug-g "Permalink to this headline")

Generate debug information for host code.

#### [4.2.3.3. --device-debug (-G)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-debug-g)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-debug-g "Permalink to this headline")

Generate debug information for device code.

If `--dopt` is not specified, then this option turns off all optimizations on device code. It is not intended for profiling; use `--generate-line-info` instead for profiling.

#### [4.2.3.4. --extensible-whole-program (-ewp)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#extensible-whole-program-ewp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#extensible-whole-program-ewp "Permalink to this headline")

Generate extensible whole program device code, which allows some calls to not be resolved until linking with libcudadevrt.

#### [4.2.3.5. --no-compress (-no-compress)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-compress-no-compress)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-compress-no-compress "Permalink to this headline")

Do not compress device code in fatbinary.

#### [4.2.3.6. --compress-mode {default|size|speed|balance|none} (-compress-mode)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compress-mode-default-size-speed-balance-none-compress-mode)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compress-mode-default-size-speed-balance-none-compress-mode "Permalink to this headline")

Choose the device code compression behavior in fatbinary.

This option is not compatible with drivers released before CUDA Toolkit’s 12.4 Release.

**Allowed Values**

`default`

> Uses the default compression mode, as if this weren’t specified. The behavior of this mode can change from version to version. It is currently equivalent to `speed`.

`size`

> Uses a compression mode more focused on reduced binary size, at the cost of compression and decompression time.

`speed`

> Uses a compression mode more focused on reduced decompression time, at the cost of less reduction in final binary size.

`balance`

> Uses a compression mode that balances binary size with compression and decompression time.

`none`

> Does not perform compression. Equivalent to `--no-compress`.

**Default Value**

`default` is used as the default mode.

#### [4.2.3.7. --relocatable-ptx (-reloc-ptx)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#relocatable-ptx-reloc-ptx)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#relocatable-ptx-reloc-ptx "Permalink to this headline")

Insert PTX from relocatable fatbins within input objects when producing final fatbin.

#### [4.2.3.8. --generate-line-info (-lineinfo)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-line-info-lineinfo)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-line-info-lineinfo "Permalink to this headline")

Generate line-number information for device code.

#### [4.2.3.9. --optimization-info kind,... (-opt-info)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#optimization-info-kind-opt-info)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optimization-info-kind-opt-info "Permalink to this headline")

Provide optimization reports for the specified kind of optimization.

The following tags are supported:

`inline`

> Emit remarks related to function inlining. Inlining pass may be invoked multiple times by the compiler and a function not inlined in an earlier pass may be inlined in a subsequent pass.

#### [4.2.3.10. --optimize level (-O)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#optimize-level-o)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optimize-level-o "Permalink to this headline")

Specify optimization level for host code.

#### [4.2.3.11. --Ofast-compile level (-Ofc)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ofast-compile-level-ofc)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ofast-compile-level-ofc "Permalink to this headline")

Specify the fast-compile level for device code, which controls the tradeoff between compilation speed and runtime performance by disabling certain optimizations at varying levels.

**Allowed Values**

- `max`: Focus only on the fastest compilation speed, disabling many optimizations.
- `mid`: Balance compile time and runtime, disabling expensive optimizations.
- `min`: More minimal impact on both compile time and runtime, minimizing some expensive optimizations.
- `0`: Disable fast-compile.

**Default Value**

The option is disabled by default.

#### [4.2.3.12. --dopt kind (-dopt)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dopt-kind-dopt)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dopt-kind-dopt "Permalink to this headline")

Enable device code optimization. When specified along with `-G`, enables limited debug information generation for optimized device code (currently, only line number information). When `-G` is not specified, `-dopt=on` is implicit.

**Allowed Values**

- `on`: enable device code optimization.

#### [4.2.3.13. --dlink-time-opt (-dlto)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dlink-time-opt-dlto)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dlink-time-opt-dlto "Permalink to this headline")

Perform link-time optimization of device code. The option ‘-lto’ is also an alias to ‘-dlto’. Link-time optimization must be specified at both compile and link time; at compile time it stores high-level intermediate code, then at link time it links together and optimizes the intermediate code. If that intermediate is not found at link time then nothing happens. Intermediate code is also stored at compile time with the `--gpu-code='lto_NN'` target. The options `-dlto -arch=sm_NN` will add a lto_NN target; if you want to only add a lto_NN target and not the compute_NN that `-arch=sm_NN` usually generates, use `-arch=lto_NN`.

#### [4.2.3.14. --gen-opt-lto (-gen-opt-lto)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gen-opt-lto-gen-opt-lto)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gen-opt-lto-gen-opt-lto "Permalink to this headline")

Run the optimizer passes before generating the LTO IR.

#### [4.2.3.15. --split-compile number (-split-compile)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#split-compile-number-split-compile)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#split-compile-number-split-compile "Permalink to this headline")

Perform compiler optimizations in parallel.

Split compilation attempts to reduce compile time by enabling the compiler to run certain optimization passes concurrently. It does this by splitting the device code into smaller translation units, each containing one or more device functions, and running optimization passes on each unit concurrently across multiple threads. It will then link back the split units prior to code generation.
The option accepts a numerical value that specifies the maximum number of threads the compiler can use. One can also allow the compiler to use the maximum threads available on the system by setting `--split-compile=0`. Setting `--split-compile=1` will cause this option to be ignored.
This option can work in conjunction with device Link Time Optimization (`-dlto`) as well as `--threads`.

#### [4.2.3.16. --split-compile-extended number (-split-compile-extended)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#split-compile-extended-number-split-compile-extended)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#split-compile-extended-number-split-compile-extended "Permalink to this headline")

A more aggressive form of `-split-compile`. Available in LTO mode only.

Extended split compilation attempts to reduce compile time even further by extending concurrent compilation through to the back-end. This agressive form of split compilation can potentially impact performance of the compiled binary.
The option accepts a numerical value that specifies the maximum number of threads the compiler can use. One can also allow the compiler to use the maximum threads available on the system by setting `--split-compile-extended=0`. Setting `--split-compile-extended=1` will cause this option to be ignored.
This option is only applicable with device Link Time Optimization (`-dlto`) and can work in conjunction with `--threads`.

#### [4.2.3.17. --jobserver (-jobserver)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#jobserver-jobserver)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#jobserver-jobserver "Permalink to this headline")

When using `-split-compile` or `--threads` inside of a build controlled by GNU Make, require that job slots are acquired Make’s jobserver for each of the threads used, helping prevent oversubscription.
This option does not restrict `-split-compile-extended` (the number of threads created by it will not be controlled).
This option only works when Make is called with *-j* set to a numerical value greater than 1, as *-j* (without a number) causes Make to skip making the jobserver and *-j1* disables all parallelism.
This requires GNU Make 4.3 or newer. For versions of Make before 4.4, or if the `--jobserver-style=pipe` is manually specified to Make, each call to NVCC must be considered a submake by make (by prepending a `+` to each line where NVCC is called) in order to provide it access to Make’s jobserver.
Using this option with an unsupported version of Make, or without the correct *-j* value may lead to undefined behavior.
We do not implement any signal handling and only minimal error handling for this feature, which can cause resources to go unused if NVCC crashes. However, it should not cause a deadlock even if an error occurs, as the job slot used by NVCC itself will always be reclaimed.

Note: This flag is only supported on Linux.

#### [4.2.3.18. --skip-ptx-semantics-check (-skip-ptx-semantics-check)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#skip-ptx-semantics-check-skip-ptx-semantics-check)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#skip-ptx-semantics-check-skip-ptx-semantics-check "Permalink to this headline")

When using `--skip-ptx-semantics-check`, the PTX semantics check step is skipped if otherwise performed. It can be used for some special or legacy code where the PTX semantics check can cause known failure.

#### [4.2.3.19. --ftemplate-backtrace-limit limit (-ftemplate-backtrace-limit)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ftemplate-backtrace-limit-limit-ftemplate-backtrace-limit)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ftemplate-backtrace-limit-limit-ftemplate-backtrace-limit "Permalink to this headline")

Set the maximum number of template instantiation notes for a single warning or error to limit.

A value of `0` is allowed, and indicates that no limit should be enforced. This value is also passed to the host compiler if it provides an equivalent flag.

#### [4.2.3.20. --ftemplate-depth limit (-ftemplate-depth)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ftemplate-depth-limit-ftemplate-depth)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ftemplate-depth-limit-ftemplate-depth "Permalink to this headline")

Set the maximum instantiation depth for template classes to limit.

This value is also passed to the host compiler if it provides an equivalent flag.

#### [4.2.3.21. --no-exceptions (-noeh)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-exceptions-noeh)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-exceptions-noeh "Permalink to this headline")

Disable exception handling for host code.

Disable exception handling for host code, by passing “-EHs-c-” (for cl.exe) and “–fno-exceptions” (for other host compilers) during host compiler invocation. These flags are added to the host compiler invocation before any flags passed directly to the host compiler with “-Xcompiler”

**Default (on Windows)**

- On Windows, `nvcc` passes /EHsc to the host compiler by default.

**Example (on Windows)**

- `nvcc --no-exceptions -Xcompiler /EHa x.cu`

#### [4.2.3.22. --shared (-shared)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#shared-shared)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#shared-shared "Permalink to this headline")

Generate a shared library during linking.

Use option `--linker-options` when other linker options are required for more control.

#### [4.2.3.23. --x {c|c++|cu} (-x)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#x-c-c-cu-x)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#x-c-c-cu-x "Permalink to this headline")

Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix.

**Allowed Values**

- `c`
- `c++`
- `cu`

**Default**

The language of the source code is determined based on the file name suffix.

#### [4.2.3.24. --std {c++03|c++11|c++14|c++17|c++20} (-std)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#std-c-03-c-11-c-14-c-17-c-20-std)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#std-c-03-c-11-c-14-c-17-c-20-std "Permalink to this headline")

Select a particular C++ dialect.

**Allowed Values**

- `c++03`
- `c++11`
- `c++14`
- `c++17`
- `c++20`

**Default**

The default C++ dialect depends on the host compiler. `nvcc` matches the default C++ dialect that the host compiler uses.

#### [4.2.3.25. --no-host-device-initializer-list (-nohdinitlist)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-host-device-initializer-list-nohdinitlist)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-host-device-initializer-list-nohdinitlist "Permalink to this headline")

Do not consider member functions of `std::initializer_list` as `__host__ __device__` functions implicitly.

#### [4.2.3.26. --expt-relaxed-constexpr (-expt-relaxed-constexpr)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#expt-relaxed-constexpr-expt-relaxed-constexpr)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#expt-relaxed-constexpr-expt-relaxed-constexpr "Permalink to this headline")

**Experimental flag**_: Allow host code to invoke ``__device__ constexpr`` functions, and device code to invoke ``__host__ constexpr`` functions._

Note that the behavior of this flag may change in future compiler releases.

#### [4.2.3.27. --extended-lambda (-extended-lambda)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#extended-lambda-extended-lambda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#extended-lambda-extended-lambda "Permalink to this headline")

Allow `__host__`, `__device__` annotations in lambda declarations.

#### [4.2.3.28. --expt-extended-lambda (-expt-extended-lambda)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#expt-extended-lambda-expt-extended-lambda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#expt-extended-lambda-expt-extended-lambda "Permalink to this headline")

Alias for `--extended-lambda`.

#### [4.2.3.29. --machine {64} (-m)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#machine-64-m)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#machine-64-m "Permalink to this headline")

Specify 64-bit architecture.

**Allowed Values**

- `64`

**Default**

This option is set based on the host platform on which `nvcc` is executed.

#### [4.2.3.30. --m64 (-m64)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#m64-m64)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#m64-m64 "Permalink to this headline")

Alias for `--machine=64`

#### [4.2.3.31. --host-linker-script {use-lcs|gen-lcs} (-hls)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#host-linker-script-use-lcs-gen-lcs-hls)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#host-linker-script-use-lcs-gen-lcs-hls "Permalink to this headline")

Use the host linker script (GNU/Linux only) to enable support for certain CUDA specific requirements, while building executable files or shared libraries.

**Allowed Values**

`use-lcs`

> Prepares a host linker script and enables host linker to support relocatable device object files that are larger in size, that would otherwise, in certain cases, cause the host linker to fail with relocation truncation error.

`gen-lcs`

> Generates a host linker script that can be passed to host linker manually, in the case where host linker is invoked separately outside of nvcc. This option can be combined with `-shared` or `-r` option to generate linker scripts that can be used while generating host shared libraries or host relocatable links respectively.
>
>
> The file generated using this options must be provided as the last input file to the host linker.
>
>
> The output is generated to stdout by default. Use the option `-o` filename to specify the output filename.

A linker script may already be in used and passed to the host linker using the host linker option `--script` (or `-T`), then the generated host linker script must augment the existing linker script. In such cases, the option `-aug-hls` must be used to generate linker script that contains only the augmentation parts. Otherwise, the host linker behaviour is undefined.

A host linker option, such as `-z` with a non-default argument, that can modify the default linker script internally, is incompatible with this option and the behavior of any such usage is undefined.

**Default Value**

`use-lcs` is used as the default type.

#### [4.2.3.32. --augment-host-linker-script (-aug-hls)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#augment-host-linker-script-aug-hls)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#augment-host-linker-script-aug-hls "Permalink to this headline")

Enables generation of host linker script that augments an existing host linker script (GNU/Linux only). See option `--host-linker-script` for more details.

#### [4.2.3.33. --relocatable-link (-r)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#relocatable-link-r)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#relocatable-link-r "Permalink to this headline")

Generates a relocatable (both host and device) object when linking.

#### [4.2.3.34. --frandom-seed (-frandom-seed)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#frandom-seed-frandom-seed)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#frandom-seed-frandom-seed "Permalink to this headline")

The user specified random seed will be used to replace random numbers used in generating symbol names and variable names. The option can be used to generate deterministicly identical ptx and object files.

If the input value is a valid number (decimal, octal, or hex), it will be used directly as the random seed.

Otherwise, the CRC value of the passed string will be used instead.

NVCC will also pass the option, as well as the user specified value to host compilers, if the host compiler is either GCC or Clang, since they support -frandom-seed option as well. Users are respoonsible for assigning different seed to different files.

### [4.2.4. Options for Passing Specific Phase Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-passing-specific-phase-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-passing-specific-phase-options "Permalink to this headline")

These flags allow for passing specific options directly to the internal compilation tools that `nvcc` encapsulates, without burdening `nvcc` with too-detailed knowledge on these tools.

#### [4.2.4.1. --compiler-options options,... (-Xcompiler)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compiler-options-options-xcompiler)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compiler-options-options-xcompiler "Permalink to this headline")

Specify options directly to the compiler/preprocessor.

#### [4.2.4.2. --linker-options options,... (-Xlinker)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#linker-options-options-xlinker)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#linker-options-options-xlinker "Permalink to this headline")

Specify options directly to the host linker.

#### [4.2.4.3. --archive-options options,... (-Xarchive)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#archive-options-options-xarchive)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#archive-options-options-xarchive "Permalink to this headline")

Specify options directly to the library manager.

#### [4.2.4.4. --ptxas-options options,... (-Xptxas)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-options-xptxas)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-options-xptxas "Permalink to this headline")

Specify options directly to `ptxas`, the PTX optimizing assembler.

#### [4.2.4.5. --nvlink-options options,... (-Xnvlink)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-options-xnvlink)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-options-xnvlink "Permalink to this headline")

Specify options directly to `nvlink`, the device linker.

### [4.2.5. Options for Guiding the Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-guiding-the-compiler-driver)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-guiding-the-compiler-driver "Permalink to this headline")

#### [4.2.5.1. --static-global-template-stub {true|false} (-static-global-template-stub)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#static-global-template-stub-true-false-static-global-template-stub)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#static-global-template-stub-true-false-static-global-template-stub "Permalink to this headline")

In whole-program compilation mode (`-rdc=false`), force `static` linkage for host side stub functions generated for `__global__` function templates.

A `__global__` function represents the entry point for GPU code execution, and is typically referenced from host code. In whole program compilation mode (`nvcc` default), the device code in each translation unit forms a self-contained device program. In the code sent to the host compiler, the CUDA frontend compiler will replace the contents of the body of the original `__global__` function or function template with calls to the CUDA runtime to launch the kernel (these are referred to as ‘stub’ functions below).

When this flag is `false`, the template stub function will have weak linkage. This causes a problem if two different translation units `a.cu` and `b.cu` have the same instatiation for a `__global__` template `G`.

For example:

```cpp
//common.h
template <typename T>
__global__ void G() { qqq = 4; }

//a.cu
static __device__ int qqq;
#include "common.h"
int main() { G<int><<<1,1>>>(); }

//b.cu
static __device__ int qqq;
#include "common.h"
int main() { G<int><<<1,1>>>(); }
```

When `a.cu` and `b.cu` are compiled in nvcc whole program mode, the device programs generated for `a.cu` and `b.cu` are separate programs, but the host linker will encounter multiple weak definitions for `G<int>` stub instantiation, and choose only one in the linked host program. As a result, launching `G<int>` from `a.cu` or `b.cu` will incorrectly launch the device program corresponding to one of `a.cu` or `b.cu`; while the correct expected behavior is that `G<int>` from `a.cu` launches the device program generated for `a.cu`, and `G<int>` from `b.cu` launches the device program generated for `b.cu`, respectively.

When the flag is `true`, the CUDA frontend compiler will make all the stub functions `static` in the generated host code. This solves the problem above, since `G<int>` in `a.cu` and `b.cu` now refer to distinct symbols in the host object code, and the host linker will not combine these symbols.

**Notes**

- This option is ignored unless the program is being compiled in whole program compilation mode (`-rdc=false`).
- Turning on this flag may break existing code in some corner cases (only in whole program compilation mode):
  1. If a `__global__` function template is declared as a friend, and the friend declaration is the first declaration of the entity.
  2. If a `__global__`  function template is referenced, but not defined in the current translation unit.

**Default**

`true`

#### [4.2.5.2. --device-entity-has-hidden-visibility {true|false} (-device-entity-has-hidden-visibility)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-entity-has-hidden-visibility-true-false-device-entity-has-hidden-visibility)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-entity-has-hidden-visibility-true-false-device-entity-has-hidden-visibility "Permalink to this headline")

This flag applies to `__global__` functions and function templates, and to `__constant__`, `__device__` and `__managed__` variables and variable templates, when using host compilers that support the `visibility` attribute (e.g. `gcc`, `clang`).

When this flag is enabled, the CUDA frontend compiler will implicitly add `__attribute__((visibility("hidden")))` to every declaration of these entities, unless the entity has internal linkage or the entity has non-default visibility e.g., due to `attribute((visibility("default")))` on an enclosing namespace.

If building a shared library, entities with `hidden` visibility cannot be referenced from outside the shared library.  This behavior is desired for `__global__` functions/template instantiations and for `__constant__/__device__/__managed__` variables and template instantiations, because the functionality of these entities depends on the CUDA Runtime (`CUDART`) library. If such entities are referenced from outside the shared library, then subtle errors can occur if a different `CUDART` is linked in to the shared library versus the user of the shared library. By forcing `hidden` visibility for such entities, these problems are avoided (the program will fail to build).

Please also see related flag `-static-global-template-stub`, which forces internal linkage for `__global__` templates in whole program compilation mode.

**Default Value**
`true`

#### [4.2.5.3. --forward-unknown-to-host-compiler (-forward-unknown-to-host-compiler)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#forward-unknown-to-host-compiler-forward-unknown-to-host-compiler)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#forward-unknown-to-host-compiler-forward-unknown-to-host-compiler "Permalink to this headline")

Forward unknown options to the host compiler. An ‘unknown option’ is a command line argument that starts with `-` followed by another character, and is not a recognized nvcc flag or an argument for a recognized nvcc flag.

If the unknown option is followed by a separate command line argument, the argument will not be forwarded, unless it begins with the `-` character.

For example:

- `nvcc -forward-unknown-to-host-compiler -foo=bar a.cu` will forward `-foo=bar` to host compiler.
- `nvcc -forward-unknown-to-host-compiler -foo bar a.cu` will report an error for `bar` argument.
- `nvcc -forward-unknown-to-host-compiler -foo -bar a.cu` will forward `-foo` and `-bar` to host compiler.

Note: On Windows, also see option `-forward-slash-prefix-opts` for forwarding options that begin with ‘/’.

#### [4.2.5.4. --forward-unknown-to-host-linker (-forward-unknown-to-host-linker)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#forward-unknown-to-host-linker-forward-unknown-to-host-linker)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#forward-unknown-to-host-linker-forward-unknown-to-host-linker "Permalink to this headline")

Forward unknown options to the host linker. An ‘unknown option’ is a command line argument that starts with `-` followed by another character, and is not a recognized nvcc flag or an argument for a recognized nvcc flag.

If the unknown option is followed by a separate command line argument, the argument will not be forwarded, unless it begins with the `-` character.

For example:

- `nvcc -forward-unknown-to-host-linker -foo=bar a.cu` will forward `-foo=bar` to host linker.
- `nvcc -forward-unknown-to-host-linker -foo bar a.cu` will report an error for `bar` argument.
- `nvcc -forward-unknown-to-host-linker -foo -bar a.cu` will forward `-foo` and `-bar` to host linker.

Note: On Windows, also see option `-forward-slash-prefix-opts` for forwarding options that begin with ‘/’.

#### [4.2.5.5. --forward-unknown-opts (-forward-unknown-opts)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#forward-unknown-opts-forward-unknown-opts)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#forward-unknown-opts-forward-unknown-opts "Permalink to this headline")

Implies the combination of options `-forward-unknown-to-host-linker` and `-forward-unknown-to-host-compiler`.

For example:

- `nvcc -forward-unknown-opts -foo=bar a.cu` will forward `-foo=bar` to the host linker and compiler.
- `nvcc -forward-unknown-opts -foo bar a.cu` will report an error for `bar` argument.
- `nvcc -forward-unknown-opts -foo -bar a.cu` will forward `-foo` and `-bar` to the host linker and compiler.

Note: On Windows, also see option `-forward-slash-prefix-opts` for forwarding options that begin with ‘/’.

#### [4.2.5.6. --forward-slash-prefix-opts (-forward-slash-prefix-opts)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#forward-slash-prefix-opts-forward-slash-prefix-opts)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#forward-slash-prefix-opts-forward-slash-prefix-opts "Permalink to this headline")

If this flag is specified, and forwarding unknown options to host toolchain is enabled (`-forward-unknown-opts` or
`-forward-unknown-to-host-linker` or `-forward-unknown-to-host-compiler`), then a command line argument beginning
with ‘/’ is  forwarded to the host toolchain.

For example:

- `nvcc -forward-slash-prefix-opts -forward-unknown-opts /T foo.cu` will forward the flag `/T` to the host compiler and linker.

When this flag is not specified, a command line argument beginning with ‘/’   is treated as an input file.

For example:

- `nvcc /T foo.cu` will treat ‘/T’ as an input file, and the Windows API function `GetFullPathName()` is used to determine the full path name.

Note: This flag is only supported on Windows.

#### [4.2.5.7. --dont-use-profile (-noprof)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dont-use-profile-noprof)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dont-use-profile-noprof "Permalink to this headline")

Do not use configurations from the `nvcc.profile` file for compilation.

#### [4.2.5.8. --threads number (-t)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#threads-number-t)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#threads-number-t "Permalink to this headline")

Specify the maximum number of threads to be used to execute the compilation steps in parallel.

This option can be used to improve the compilation speed when compiling for multiple architectures. The compiler creates _number_ threads to execute the compilation steps in parallel. If _number_ is 1, this option is ignored. If _number_ is 0, the number of threads used is the number of CPUs on the machine.

#### [4.2.5.9. --dryrun (-dryrun)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dryrun-dryrun)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dryrun-dryrun "Permalink to this headline")

List the compilation sub-commands without executing them.

#### [4.2.5.10. --verbose (-v)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#verbose-v)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#verbose-v "Permalink to this headline")

List the compilation sub-commands while executing them.

#### [4.2.5.11. --keep (-keep)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#keep-keep)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keep-keep "Permalink to this headline")

Keep all intermediate files that are generated during internal compilation steps.

#### [4.2.5.12. --keep-dir directory (-keep-dir)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#keep-dir-directory-keep-dir)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keep-dir-directory-keep-dir "Permalink to this headline")

Keep all intermediate files that are generated during internal compilation steps in this directory.

#### [4.2.5.13. --save-temps (-save-temps)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#save-temps-save-temps)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#save-temps-save-temps "Permalink to this headline")

This option is an alias of `--keep`.

#### [4.2.5.14. --clean-targets (-clean)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#clean-targets-clean)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#clean-targets-clean "Permalink to this headline")

Delete all the non-temporary files that the same `nvcc` command would generate without this option.

This option reverses the behavior of `nvcc`. When specified, none of the compilation phases will be executed. Instead, all of the non-temporary files that `nvcc` would otherwise create will be deleted.

#### [4.2.5.15. --run-args arguments,... (-run-args)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#run-args-arguments-run-args)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#run-args-arguments-run-args "Permalink to this headline")

Specify command line arguments for the executable when used in conjunction with `--run`.

#### [4.2.5.16. --use-local-env (-use-local-env)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#use-local-env-use-local-env)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#use-local-env-use-local-env "Permalink to this headline")

Use this flag to force nvcc to assume that the environment for cl.exe has already been set up, and skip running the
batch file from the MSVC installation that sets up the environment for cl.exe. This can significantly reduce overall
compile time for small programs.

#### [4.2.5.17. --force-cl-env-setup (-force-cl-env-setup)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#force-cl-env-setup-force-cl-env-setup)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#force-cl-env-setup-force-cl-env-setup "Permalink to this headline")

Force nvcc to always run the batch file from the MSVC installation to set up the environment for cl.exe
(matching legacy nvcc behavior).

If this flag is not specified, by default, nvcc will skip running the batch file if the following conditions are
satisfied : cl.exe is in the PATH, environment variable VSCMD_VER is set, and, if `-ccbin` is specifed, then compiler
denoted by `-ccbin` matches the cl.exe in the PATH. Skipping the batch file execution can reduce overall compile time
significantly for small programs.

#### [4.2.5.18. --input-drive-prefix prefix (-idp)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#input-drive-prefix-prefix-idp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#input-drive-prefix-prefix-idp "Permalink to this headline")

Specify the input drive prefix.

On Windows, all command line arguments that refer to file names must be converted to the Windows native format before they are passed to pure Windows executables. This option specifies how the current development environment represents absolute paths. Use `/cygwin/` as `prefix` for Cygwin build environments and `/` as `prefix` for MinGW.

#### [4.2.5.19. --dependency-drive-prefix prefix (-ddp)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dependency-drive-prefix-prefix-ddp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dependency-drive-prefix-prefix-ddp "Permalink to this headline")

Specify the dependency drive prefix.

On Windows, when generating dependency files (see `--generate-dependencies`), all file names must be converted appropriately for the instance of `make` that is used. Some instances of `make` have trouble with the colon in absolute paths in the native Windows format, which depends on the environment in which the `make` instance has been compiled. Use `/cygwin/` as `prefix` for a Cygwin `make`, and `/` as `prefix` for MinGW. Or leave these file names in the native Windows format by specifying nothing.

#### [4.2.5.20. --drive-prefix prefix (-dp)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#drive-prefix-prefix-dp)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#drive-prefix-prefix-dp "Permalink to this headline")

Specify the drive prefix.

This option specifies `prefix` as both `--input-drive-prefix` and `--dependency-drive-prefix`.

#### [4.2.5.21. --dependency-target-name target (-MT)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dependency-target-name-target-mt)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dependency-target-name-target-mt "Permalink to this headline")

Specify the target name of the generated rule when generating a dependency file (see `--generate-dependencies`).

#### [4.2.5.22. --no-align-double](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-align-double)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-align-double "Permalink to this headline")

Specify that `-malign-double` should not be passed as a compiler argument on 32-bit platforms.

**WARNING:** this makes the ABI incompatible with the CUDA’s kernel ABI for certain 64-bit types.

#### [4.2.5.23. --no-device-link (-nodlink)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-device-link-nodlink)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-device-link-nodlink "Permalink to this headline")

Skip the device link step when linking object files.

#### [4.2.5.24. --allow-unsupported-compiler (-allow-unsupported-compiler)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-guiding-compiler-driver-allow-unsupported-compiler)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-guiding-compiler-driver-allow-unsupported-compiler "Permalink to this headline")

Disable nvcc check for supported host compiler versions.

Using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk. This option has no effect on MacOS.

### [4.2.6. Options for Steering CUDA Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-steering-cuda-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-cuda-compilation "Permalink to this headline")

#### [4.2.6.1. --default-stream {legacy|null|per-thread} (-default-stream)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#default-stream-legacy-null-per-thread-default-stream)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#default-stream-legacy-null-per-thread-default-stream "Permalink to this headline")

Specify the stream that CUDA commands from the compiled program will be sent to by default.

**Allowed Values**

`legacy`

> The CUDA legacy stream (per context, implicitly synchronizes with other streams)

`per-thread`

> Normal CUDA stream (per thread, does not implicitly synchronize with other streams)

`null`

> Deprecated alias for `legacy`

**Default**

`legacy` is used as the default stream.

### [4.2.7. Options for Steering GPU Code Generation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-for-steering-gpu-code-generation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation "Permalink to this headline")

#### [4.2.7.1. --gpu-architecture (-arch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-architecture-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-architecture-arch "Permalink to this headline")

Specify the name of the class of NVIDIA virtual GPU architecture for which the CUDA input files must be compiled.

With the exception as described for the shorthand below, the architecture specified with this option must be a _virtual_ architecture (such as compute_100). Normally, this option alone does not trigger assembly of the generated PTX for a _real_ architecture (that is the role of `nvcc` option `--gpu-code`, see below); rather, its purpose is to control preprocessing and compilation of the input to PTX.

For convenience, in case of simple `nvcc` compilations, the following shorthand is supported. If no value for option `--gpu-code` is specified, then the value of this option defaults to the value of `--gpu-architecture`. In this situation, as the only exception to the description above, the value specified for `--gpu-architecture` may be a _real_ architecture (such as a sm_100), in which case `nvcc` uses the specified _real_ architecture and its closest _virtual_ architecture as the effective architecture values. For example, `nvcc --gpu-architecture=sm_100` is equivalent to `nvcc --gpu-architecture=compute_100 --gpu-code=sm_100,compute_100`. If the architecture-specific _real_ gpu (such as `-arch=sm_90a`) is specified, then both architecture-specific and non-architecture-specific virtual code are added to the code list: `--gpu-architecture=compute_90a --gpu-code=sm_90a,compute_90,compute_90a`.

When `-arch=native` is specified, `nvcc` detects the visible GPUs on the system and generates codes for them, no PTX program will be generated for this option. It is a warning if there are no visible supported GPU on the system, and the default architecture will be used.

If `-arch=all` is specified, `nvcc` embeds a compiled code image for all supported architectures `(sm_*)`, and a PTX program for the highest major virtual architecture. For `-arch=all-major`, `nvcc` embeds a compiled code image for all supported major versions `(sm_*0)`, plus the earliest supported, and adds a PTX program for the highest major virtual architecture.

See [Virtual Architecture Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list) for the list of supported _virtual_ architectures and [GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list) for the list of supported _real_ architectures.

**Default**

`sm_75` is used as the default value; PTX is generated for `compute_75` then assembled and optimized for `sm_75`.

#### [4.2.7.2. --gpu-code code,... (-code)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-code-code-code)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-code-code-code "Permalink to this headline")

Specify the name of the NVIDIA GPU to assemble and optimize PTX for.

`nvcc` embeds a compiled code image in the resulting executable for each specified _code_ architecture, which is a true binary load image for each _real_ architecture (such as sm_100), and PTX code for the _virtual_ architecture (such as compute_100).

During runtime, such embedded PTX code is dynamically compiled by the CUDA runtime system if no binary load image is found for the _current_ GPU.

Architectures specified for options `--gpu-architecture` and `--gpu-code` may be _virtual_ as well as _real_, but the `code` architectures must be compatible with the `arch` architecture. When the `--gpu-code` option is used, the value for the `--gpu-architecture` option must be a _virtual_ PTX architecture.

For instance, `--gpu-architecture=compute_100` is not compatible with `--gpu-code=sm_90`, because the earlier compilation stages will assume the availability of `compute_100` features that are not present on `sm_90`.

See [Virtual Architecture Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list) for the list of supported _virtual_ architectures and [GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list) for the list of supported _real_ architectures.

#### [4.2.7.3. --generate-code specification (-gencode)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generate-code-specification-gencode)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generate-code-specification-gencode "Permalink to this headline")

This option provides a generalization of the `--gpu-architecture=arch --gpu-code=code,...` option combination for specifying `nvcc` behavior with respect to code generation.

Where use of the previous options generates code for different _real_ architectures with the PTX for the same _virtual_ architecture, option `--generate-code` allows multiple PTX generations for different _virtual_ architectures. In fact, `--gpu-architecture=arch --gpu-code=code,...` is equivalent to `--generate-code=arch=arch,code=code,...`.

`--generate-code` options may be repeated for different virtual architectures.

See [Virtual Architecture Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list) for the list of supported _virtual_ architectures and [GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list) for the list of supported _real_ architectures.

#### [4.2.7.4. --relocatable-device-code {true|false} (-rdc)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#relocatable-device-code-true-false-rdc)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#relocatable-device-code-true-false-rdc "Permalink to this headline")

Enable or disable the generation of relocatable device code.

If disabled, executable device code is generated. Relocatable device code must be linked before it can be executed.

**Allowed Values**

- `true`
- `false`

**Default**

The generation of relocatable device code is disabled.

#### [4.2.7.5. --entries entry,... (-e)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#entries-entry-e)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#entries-entry-e "Permalink to this headline")

Specify the global entry functions for which code must be generated.

PTX generated for all entry functions, but only the selected entry functions are assembled. Entry function names for this option must be specified in the mangled name.

**Default**

`nvcc` generates code for all entry functions.

#### [4.2.7.6. --maxrregcount amount (-maxrregcount)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#maxrregcount-amount-maxrregcount)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#maxrregcount-amount-maxrregcount "Permalink to this headline")

Specify the maximum amount of registers that GPU functions can use.

Until a function-specific limit, a higher value will generally increase the performance of individual GPU threads that execute this function. However, because thread registers are allocated from a global register pool on each GPU, a higher value of this option will also reduce the maximum thread block size, thereby reducing the amount of thread parallelism. Hence, a good `maxrregcount` value is the result of a trade-off.

A value less than the minimum registers required by ABI will be bumped up by the compiler to ABI minimum limit.

User program may not be able to make use of all registers as some registers are reserved by compiler.

**Default**

No maximum is assumed.

#### [4.2.7.7. --use_fast_math (-use_fast_math)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#use-fast-math-use-fast-math)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#use-fast-math-use-fast-math "Permalink to this headline")

Make use of fast math library.

`--use_fast_math` implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`.

#### [4.2.7.8. --ftz {true|false} (-ftz)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ftz-true-false-ftz)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ftz-true-false-ftz "Permalink to this headline")

Control single-precision denormals support.

`--ftz=true` flushes denormal values to zero and `--ftz=false` preserves denormal values.

`--use_fast_math` implies `--ftz=true`.

**Allowed Values**

- `true`
- `false`

**Default**

This option is set to `false` and `nvcc` preserves denormal values.

#### [4.2.7.9. --prec-div {true|false} (-prec-div)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#prec-div-true-false-prec-div)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#prec-div-true-false-prec-div "Permalink to this headline")

This option controls single-precision floating-point division and reciprocals.

`--prec-div=true` enables the IEEE round-to-nearest mode and `--prec-div=false` enables the fast approximation mode.

`--use_fast_math` implies `--prec-div=false`.

**Allowed Values**

- `true`
- `false`

**Default**

This option is set to `true` and `nvcc` enables the IEEE round-to-nearest mode.

#### [4.2.7.10. --prec-sqrt {true|false} (-prec-sqrt)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#prec-sqrt-true-false-prec-sqrt)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#prec-sqrt-true-false-prec-sqrt "Permalink to this headline")

This option controls single-precision floating-point square root.

`--prec-sqrt=true` enables the IEEE round-to-nearest mode and `--prec-sqrt=false` enables the fast approximation mode.

`--use_fast_math` implies `--prec-sqrt=false`.

**Allowed Values**

- `true`
- `false`

**Default**

This option is set to `true` and `nvcc` enables the IEEE round-to-nearest mode.

#### [4.2.7.11. --fmad {true|false} (-fmad)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fmad-true-false-fmad)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fmad-true-false-fmad "Permalink to this headline")

This option enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA).

`--use_fast_math` implies `--fmad=true`.

**Allowed Values**

- `true`
- `false`

**Default**

This option is set to `true` and `nvcc` enables the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA).

#### [4.2.7.12. --extra-device-vectorization (-extra-device-vectorization)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#extra-device-vectorization-extra-device-vectorization)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#extra-device-vectorization-extra-device-vectorization "Permalink to this headline")

This option enables more aggressive device code vectorization.

#### [4.2.7.13. --compile-as-tools-patch (-astoolspatch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compile-as-tools-patch-astoolspatch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compile-as-tools-patch-astoolspatch "Permalink to this headline")

Compile patch code for CUDA tools. Implies [–keep-device-functions](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation-keep-device-functions).

May only be used in conjunction with `--ptx` or `--cubin` or `--fatbin`.

Shall not be used in conjunction with `-rdc=true` or `-ewp`.

Some PTX ISA features may not be usable in this compilation mode.

#### [4.2.7.14. --keep-device-functions (-keep-device-functions)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#keep-device-functions-keep-device-functions)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keep-device-functions-keep-device-functions "Permalink to this headline")

In whole program compilation mode, preserve user defined external linkage `__device__` function definitions in generated PTX.

#### [4.2.7.15. --jump-table-density percentage (-jtd)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#jump-table-density-percentage-jtd)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#jump-table-density-percentage-jtd "Permalink to this headline")

Specify the case density percentage in switch statements, and use it as a minimal threshold to determine whether jump table(brx.idx instruction) will be used to implement a switch statement.

The percentage ranges from 0 to 101 inclusively.

**Default**

This option is set to `101` and `nvcc` disables jump table generation for switch statements.

### [4.2.8. Generic Tool Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#generic-tool-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generic-tool-options "Permalink to this headline")

#### [4.2.8.1. --disable-warnings (-w)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#disable-warnings-w)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#disable-warnings-w "Permalink to this headline")

Inhibit all warning messages.

#### [4.2.8.2. --source-in-ptx (-src-in-ptx)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#source-in-ptx-src-in-ptx)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#source-in-ptx-src-in-ptx "Permalink to this headline")

Interleave source in PTX.

May only be used in conjunction with `--device-debug` or `--generate-line-info`.

#### [4.2.8.3. --restrict (-restrict)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#restrict-restrict)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#restrict-restrict "Permalink to this headline")

Assert that all kernel pointer parameters are restrict pointers.

#### [4.2.8.4. --Wno-deprecated-gpu-targets (-Wno-deprecated-gpu-targets)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wno-deprecated-gpu-targets-wno-deprecated-gpu-targets)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wno-deprecated-gpu-targets-wno-deprecated-gpu-targets "Permalink to this headline")

Suppress warnings about deprecated GPU target architectures.

#### [4.2.8.5. --Wno-deprecated-declarations (-Wno-deprecated-declarations)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wno-deprecated-declarations-wno-deprecated-declarations)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wno-deprecated-declarations-wno-deprecated-declarations "Permalink to this headline")

Suppress warning on use of a deprecated entity.

#### [4.2.8.6. --Wreorder (-Wreorder)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wreorder-wreorder)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wreorder-wreorder "Permalink to this headline")

Generate warnings when member initializers are reordered.

#### [4.2.8.7. --Wdefault-stream-launch (-Wdefault-stream-launch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wdefault-stream-launch-wdefault-stream-launch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wdefault-stream-launch-wdefault-stream-launch "Permalink to this headline")

Generate warning when an explicit stream argument is not provided in the `<<<...>>>` kernel launch syntax.

#### [4.2.8.8. --Wmissing-launch-bounds (-Wmissing-launch-bounds)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wmissing-launch-bounds-wmissing-launch-bounds)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wmissing-launch-bounds-wmissing-launch-bounds "Permalink to this headline")

Generate warning when a `__global__` function does not have an explicit `__launch_bounds__` annotation.

#### [4.2.8.9. --Wext-lambda-captures-this (-Wext-lambda-captures-this)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#wext-lambda-captures-this-wext-lambda-captures-this)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#wext-lambda-captures-this-wext-lambda-captures-this "Permalink to this headline")

Generate warning when an extended lambda implicitly captures `this`.

#### [4.2.8.10. --Werror kind,... (-Werror)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#werror-kind-werror)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#werror-kind-werror "Permalink to this headline")

Make warnings of the specified kinds into errors.

The following is the list of warning kinds accepted by this option:

`all-warnings`

> Treat all warnings as errors.

`cross-execution-space-call`

> Be more strict about unsupported cross execution space calls. The compiler will generate an error instead of a warning for a call from a `__host__``__device__` to a `__host__` function.

`reorder`

> Generate errors when member initializers are reordered.

`default-stream-launch`

> Generate error when an explicit stream argument is not provided in the `<<<...>>>` kernel launch syntax.

`missing-launch-bounds`

> Generate warning when a `__global__` function does not have an explicit `__launch_bounds__` annotation.

`ext-lambda-captures-this`

> Generate error when an extended lambda implicitly captures `this`.

`deprecated-declarations`

> Generate error on use of a deprecated entity.

#### [4.2.8.11. --display-error-number (-err-no)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#display-error-number-err-no)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#display-error-number-err-no "Permalink to this headline")

This option displays a diagnostic number for any message generated by the CUDA frontend compiler (note: not the host compiler).

#### [4.2.8.12. --no-display-error-number (-no-err-no)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#no-display-error-number-no-err-no)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#no-display-error-number-no-err-no "Permalink to this headline")

This option disables the display of a diagnostic number for any message generated by the CUDA frontend compiler (note: not the host compiler).

#### [4.2.8.13. --diag-error errNum,... (-diag-error)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#diag-error-errnum-diag-error)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#diag-error-errnum-diag-error "Permalink to this headline")

Emit error for specified diagnostic message(s) generated by the CUDA frontend compiler (note: does not affect diagnostics generated by the host compiler/preprocessor).

#### [4.2.8.14. --diag-suppress errNum,... (-diag-suppress)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#diag-suppress-errnum-diag-suppress)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#diag-suppress-errnum-diag-suppress "Permalink to this headline")

Suppress specified diagnostic message(s) generated by the CUDA frontend compiler (note: does not affect diagnostics generated by the host compiler/preprocessor).

#### [4.2.8.15. --diag-warn errNum,... (-diag-warn)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#diag-warn-errnum-diag-warn)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#diag-warn-errnum-diag-warn "Permalink to this headline")

Emit warning for specified diagnostic message(s) generated by the CUDA frontend compiler (note: does not affect diagnostics generated by the host compiler/preprocessor).

#### [4.2.8.16. --resource-usage (-res-usage)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#resource-usage-res-usage)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#resource-usage-res-usage "Permalink to this headline")

Show resource usage such as registers and memory of the GPU code.

This option implies `--nvlink-options=--verbose` when `--relocatable-device-code=true` is set. Otherwise, it implies `--ptxas-options=--verbose`.

#### [4.2.8.17. --device-stack-protector {true|false} (-device-stack-protector)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-stack-protector-true-false-device-stack-protector)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-stack-protector-true-false-device-stack-protector "Permalink to this headline")

Enable or disable the generation of stack canaries in device code.

Stack canaries make it more difficult to exploit certain types of memory safety bugs involving stack-local variables.
The compiler uses heuristics to assess the risk of such a bug in each function.  Only those functions which are deemed high-risk make use of a stack canary.

**Allowed Values**

- `true`
- `false`

**Default**

The generation of stack canaries in device code is disabled.

#### [4.2.8.18. --help (-h)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#help-h)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#help-h "Permalink to this headline")

Print help information on this tool.

#### [4.2.8.19. --version (-V)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#version-v)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#version-v "Permalink to this headline")

Print version information on this tool.

#### [4.2.8.20. --options-file file,... (-optf)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-file-file-optf)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-file-file-optf "Permalink to this headline")

Include command line options from specified file.

#### [4.2.8.21. --time  filename (-time)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#time-filename-time)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#time-filename-time "Permalink to this headline")

Generate a comma separated value table with the time taken by each compilation phase, and append it at the end of the file given as the option argument. If the file is empty, the column headings are generated in the first row of the table.

If the file name is `-`, the timing data is generated in stdout.

#### [4.2.8.22. --qpp-config config (-qpp-config)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#qpp-config-config-qpp-config)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#qpp-config-config-qpp-config "Permalink to this headline")

Specify the configuration ([[compiler/]version,][target]) when using q++ host compiler. The argument will be forwarded to the q++ compiler with its -V flag.

#### [4.2.8.23. --list-gpu-code (-code-ls)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#list-gpu-code-code-ls)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#list-gpu-code-code-ls "Permalink to this headline")

List the non-architecture-specific gpu architectures (sm_XX) supported by the tool and exit.

If both [–list-gpu-code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generic-tool-options-list-gpu-code) and [–list-gpu-arch](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generic-tool-options-list-gpu-arch) are set, the list is displayed using the same format as the [–generate-code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation) value.

#### [4.2.8.24. --list-gpu-arch (-arch-ls)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#list-gpu-arch-arch-ls)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#list-gpu-arch-arch-ls "Permalink to this headline")

List the non-architecture-specific virtual device architectures (compute_XX) supported by the tool and exit.

If both [–list-gpu-arch](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generic-tool-options-list-gpu-arch) and [–list-gpu-code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#generic-tool-options-list-gpu-code) are set, the list is displayed using the same format as the [–generate-code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation) value.

#### [4.2.8.25. --fdevice-time-trace (-fdevice-time-trace)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fdevice-time-trace-fdevice-time-trace)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fdevice-time-trace-fdevice-time-trace "Permalink to this headline")

Enables the time profiler, outputting a JSON file based on given file name. If file name is ‘-’, the JSON file will have the same name as the user provided output file [-o](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#file-and-path-specifications-output-file), otherwise it will be set to ‘trace.json’.

#### [4.2.8.26. --fdevice-sanitize (-fdevice-sanitize)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fdevice-sanitize-fdevice-sanitize)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fdevice-sanitize-fdevice-sanitize "Permalink to this headline")

Enable compiler instrumentation for the compute-sanitizer tool specified by the option.
Currently, only the value `memcheck` (memory check) is supported. Binary needs to be
executed using `compute-sanitizer` at runtime, otherwise behavior is undefined.
For more informations about compile time instrumentation and the compute sanitizer,
consult the [corresponding documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer#compile-time-patching).

### [4.2.9. Phase Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#phase-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#phase-options "Permalink to this headline")

The following sections lists some useful options to lower level compilation tools.

#### [4.2.9.1. Ptxas Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options "Permalink to this headline")

The following table lists some useful `ptxas` options which can be specified with `nvcc` option `-Xptxas`.

##### [4.2.9.1.1. --allow-expensive-optimizations (-allow-expensive-optimizations)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#allow-expensive-optimizations-allow-expensive-optimizations)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#allow-expensive-optimizations-allow-expensive-optimizations "Permalink to this headline")

Enable (disable) to allow compiler to perform expensive optimizations using maximum available resources (memory and compile-time).

If unspecified, default behavior is to enable this feature for optimization level >= `O2`.

##### [4.2.9.1.2. --compile-only (-c)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#compile-only-c)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compile-only-c "Permalink to this headline")

Generate relocatable object.

##### [4.2.9.1.3. --def-load-cache (-dlcm)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#def-load-cache-dlcm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#def-load-cache-dlcm "Permalink to this headline")

Default cache modifier on global/generic load.

##### [4.2.9.1.4. --def-store-cache (-dscm)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#def-store-cache-dscm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#def-store-cache-dscm "Permalink to this headline")

Default cache modifier on global/generic store.

##### [4.2.9.1.5. --device-debug (-g)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-device-debug)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-device-debug "Permalink to this headline")

Semantics same as `nvcc` option `--device-debug`.

##### [4.2.9.1.6. --disable-optimizer-constants (-disable-optimizer-consts)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#disable-optimizer-constants-disable-optimizer-consts)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#disable-optimizer-constants-disable-optimizer-consts "Permalink to this headline")

Disable use of optimizer constant bank.

##### [4.2.9.1.7. --entry entry,... (-e)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#entry-entry-e)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#entry-entry-e "Permalink to this headline")

Semantics same as `nvcc` option `--entries`.

##### [4.2.9.1.8. --fmad (-fmad)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fmad-fmad)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fmad-fmad "Permalink to this headline")

Semantics same as `nvcc` option `--fmad`.

##### [4.2.9.1.9. --force-load-cache (-flcm)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#force-load-cache-flcm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#force-load-cache-flcm "Permalink to this headline")

Force specified cache modifier on global/generic load.

##### [4.2.9.1.10. --force-store-cache (-fscm)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#force-store-cache-fscm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#force-store-cache-fscm "Permalink to this headline")

Force specified cache modifier on global/generic store.

##### [4.2.9.1.11. --generate-line-info (-lineinfo)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-generate-line-info)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-generate-line-info "Permalink to this headline")

Semantics same as `nvcc` option `--generate-line-info`.

##### [4.2.9.1.12. --gpu-name gpuname (-arch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-name-gpuname-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-name-gpuname-arch "Permalink to this headline")

Specify name of NVIDIA GPU to generate code for.

This option also takes virtual compute architectures, in which case code generation is suppressed.
This can be used for parsing only.

PTX for .target sm_XY can be compiled to all GPU targets sm_MN, sm_MNa, SM_MNf where MN >= XY. PTX
for .target sm_XYf can be compiled to GPU targets sm_XZ, sm_XZf, sm_XZa where Z >= Y and sm_XY and
sm_XZ belong in same family. PTX with .target sm_XYa can only be compiled to GPU target sm_XYa.

**Allowed Values**

> | `compute_75` | `compute_80` | `compute_86` | `compute_87` |
> | --- | --- | --- | --- |
> | `compute_88` | `compute_89` | `compute_90` | `compute_90a` |
> | `compute_100` | `compute_100f` | `compute_100a` | `compute_103` |
> | `compute_103f` | `compute_103a` | `compute_110` | `compute_110f` |
> | `compute_110a` | `compute_120` | `compute_120f` | `compute_120a` |
> | `compute_121` | `compute_121f` | `compute_121a` |  |
> | `sm_75` | `sm_80` | `sm_86` | `sm_87` |
> | `sm_88` | `sm_89` | `sm_90` | `sm_90a` |
> | `sm_100` | `sm_100f` | `sm_100a` | `sm_103` |
> | `sm_103f` | `sm_103a` | `sm_110` | `sm_110f` |
> | `sm_110a` | `sm_120` | `sm_120f` | `sm_120a` |
> | `sm_121` | `sm_121f` | `sm_121a` |  |

Default value: `sm_75`

##### [4.2.9.1.13. --help (-h)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-help)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-help "Permalink to this headline")

Semantics same as `nvcc` option `--help`.

##### [4.2.9.1.14. --machine (-m)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#machine-m)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#machine-m "Permalink to this headline")

Semantics same as `nvcc` option `--machine`.

##### [4.2.9.1.15. --maxrregcount amount (-maxrregcount)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-maxrregcount)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-maxrregcount "Permalink to this headline")

Semantics same as `nvcc` option `--maxrregcount`.

##### [4.2.9.1.16. --opt-level N (-O)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#opt-level-n-o)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#opt-level-n-o "Permalink to this headline")

Specify optimization level.

Default value: `3`.

##### [4.2.9.1.17. --options-file file,... (-optf)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-options-file)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-options-file "Permalink to this headline")

Semantics same as `nvcc` option `--options-file`.

##### [4.2.9.1.18. --position-independent-code (-pic)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#position-independent-code-pic)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#position-independent-code-pic "Permalink to this headline")

Generate position-independent code.

Default value:

For whole-program compilation: `true`

Otherwise: `false`

##### [4.2.9.1.19. --preserve-relocs (-preserve-relocs)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#preserve-relocs-preserve-relocs)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#preserve-relocs-preserve-relocs "Permalink to this headline")

This option will make `ptxas` to generate relocatable references for variables and preserve relocations generated for them in linked executable.

##### [4.2.9.1.20. --sp-bounds-check (-sp-bounds-check)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#sp-bounds-check-sp-bounds-check)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#sp-bounds-check-sp-bounds-check "Permalink to this headline")

Generate stack-pointer bounds-checking code sequence.

This option is turned on automatically when `--device-debug` or `--opt-level=0` is specified.

##### [4.2.9.1.21. --suppress-async-bulk-multicast-advisory-warning (-suppress-async-bulk-multicast-advisory-warning)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#suppress-async-bulk-multicast-advisory-warning-suppress-async-bulk-multicast-advisory-warning)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#suppress-async-bulk-multicast-advisory-warning-suppress-async-bulk-multicast-advisory-warning "Permalink to this headline")

Suppress the warning on use of `.multicast::cluster` modifier on `cp.async.bulk{.tensor}` instruction with `sm_90`.

##### [4.2.9.1.22. --verbose (-v)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-verbose)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-verbose "Permalink to this headline")

Enable verbose mode which prints code generation statistics.

##### [4.2.9.1.23. --version (-V)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-version)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-version "Permalink to this headline")

Semantics same as `nvcc` option `--version`.

##### [4.2.9.1.24. --warning-as-error (-Werror)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#warning-as-error-werror)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#warning-as-error-werror "Permalink to this headline")

Make all warnings into errors.

##### [4.2.9.1.25. --warn-on-double-precision-use (-warn-double-usage)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#warn-on-double-precision-use-warn-double-usage)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#warn-on-double-precision-use-warn-double-usage "Permalink to this headline")

Warning if double(s) are used in an instruction.

##### [4.2.9.1.26. --warn-on-local-memory-usage (-warn-lmem-usage)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#warn-on-local-memory-usage-warn-lmem-usage)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#warn-on-local-memory-usage-warn-lmem-usage "Permalink to this headline")

Warning if local memory is used.

##### [4.2.9.1.27. --warn-on-spills (-warn-spills)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#warn-on-spills-warn-spills)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#warn-on-spills-warn-spills "Permalink to this headline")

Warning if registers are spilled to local memory.

##### [4.2.9.1.28. --compile-as-tools-patch (-astoolspatch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-compile-as-tools-patch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-compile-as-tools-patch "Permalink to this headline")

Compile patch code for CUDA tools.

Shall not be used in conjunction with `-Xptxas -c` or `-ewp`.

Some PTX ISA features may not be usable in this compilation mode.

##### [4.2.9.1.29. --maxntid (-maxntid)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#maxntid-maxntid)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#maxntid-maxntid "Permalink to this headline")

Specify the maximum number of threads that a thread block can have.

This option will be ignored if used along with `-maxrregcount` option. This option is also ignored
for entry functions that have `.maxntid` directive specified.

##### [4.2.9.1.30. --minnctapersm (-minnctapersm)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#minnctapersm-minnctapersm)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#minnctapersm-minnctapersm "Permalink to this headline")

Specify the minimum number of CTAs to be mapped to an SM.

This option will be ignored if used along with `-maxrregcount` option. This option is also ignored
for entry functions that have `.minnctapersm` directive specified.

##### [4.2.9.1.31. --override-directive-values (-override-directive-values)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#override-directive-values-override-directive-values)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#override-directive-values-override-directive-values "Permalink to this headline")

Override the PTX directives values by the corresponding option values.

This option is effective only for `-minnctapersm`, `-maxntid` and `-maxrregcount` options.

##### [4.2.9.1.32. --make-errors-visible-at-exit (-make-errors-visible-at-exit)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#make-errors-visible-at-exit-make-errors-visible-at-exit)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#make-errors-visible-at-exit-make-errors-visible-at-exit "Permalink to this headline")

Generate required instructions at exit point to make memory faults and errors visible at exit.

##### [4.2.9.1.33. --Ofast-compile level (-Ofc)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-ofast-compile)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-ofast-compile "Permalink to this headline")

Specify the fast-compile level for device code, which controls the tradeoff between compilation speed and runtime performance by disabling certain optimizations at varying levels. Passed automatically at the same level when nvcc is called with `--Ofast-compile`.

**Allowed Values**

- `max`: Focus only on the fastest compilation speed, disabling many optimizations.
- `mid`: Balance compile time and runtime, disabling expensive optimizations.
- `min`: More minimal impact on both compile time and runtime, minimizing some expensive optimizations.
- `0`: Disable fast-compile.

**Default Value**

The option is disabled by default.

##### [4.2.9.1.34. --device-stack-protector (-device-stack-protector)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-stack-protector-device-stack-protector)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-stack-protector-device-stack-protector "Permalink to this headline")

Enable or disable the generation of stack canaries in device code.

Stack canaries make it more difficult to exploit certain types of memory safety bugs involving stack-local variables.
The compiler uses heuristics to assess the risk of such a bug in each function.  Only those functions which are deemed high-risk make use of a stack canary.

**Allowed Values**

- `true`
- `false`

**Default**

The generation of stack canaries in device code is disabled.

##### [4.2.9.1.35. --g-tensor-memory-access-check (-g-tmem-access-check)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#g-tensor-memory-access-check-g-tmem-access-check)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#g-tensor-memory-access-check-g-tmem-access-check "Permalink to this headline")

Enable tensor memory access checks for `tcgen05` operations.

This option is enabled by default with `--device-debug` (`-g`) option.

##### [4.2.9.1.36. --gno-tensor-memory-access-check (-gno-tmem-access-check)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gno-tensor-memory-access-check-gno-tmem-access-check)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gno-tensor-memory-access-check-gno-tmem-access-check "Permalink to this headline")

Disable tensor memory access checks for `tcgen05` operations.

This option will override the `--g-tensor-memory-access-check` option if both are specified.

##### [4.2.9.1.37. --split-compile (-split-compile)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#split-compile-split-compile)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#split-compile-split-compile "Permalink to this headline")

Specify the maximum amount of concurrent threads to be utilized when running compiler optimizations.

If value specified is `1`, option will be ignored.
If value specified is `0`, then the number of threads will be the number of CPUs on
the underlying machine.

##### [4.2.9.1.38. --jobserver (-jobserver)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ptxas-options-jobserver)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options-jobserver "Permalink to this headline")

Enable GNU jobserver support.

When using `-split-compile` inside of a build controlled by GNU Make,
require that job slots are acquired Make’s jobserver for each of the threads used, helping
prevent oversubscription.
This option only works when Make is called with `-j` set to a numerical value greater
than 1, as `-j` (without a number) causes Make to skip making the jobserver and
`-j1` disables all parallelism.
This requires GNU Make 4.3 or newer.
Using this option with an unsupported version of Make, or without the correct `-j`
value may lead to undefined behavior.

Note: This flag is only supported on Linux.

#### [4.2.9.2. --sanitize (-sanitize)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#sanitize-sanitize)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#sanitize-sanitize "Permalink to this headline")

Enable compiler instrumentation for the compute-sanitizer tool specified by the option.
Currently, only the value `memcheck` (memory check) is supported. Binary needs to be
executed using `compute-sanitizer` at runtime, otherwise behavior is undefined.
For more informations about compile time instrumentation and the compute sanitizer,
consult the [corresponding documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer#compile-time-patching).

#### [4.2.9.3. NVLINK Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options "Permalink to this headline")

The following is a list of some useful `nvlink` options which can be specified with `nvcc` option `--nvlink-options`.

##### [4.2.9.3.1. --disable-warnings (-w)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-disable-warnings)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-disable-warnings "Permalink to this headline")

Inhibit all warning messages.

##### [4.2.9.3.2. --preserve-relocs (-preserve-relocs)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-preserve-relocs)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-preserve-relocs "Permalink to this headline")

Preserve resolved relocations in linked executable.

##### [4.2.9.3.3. --verbose (-v)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-verbose)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-verbose "Permalink to this headline")

Enable verbose mode which prints code generation statistics.

##### [4.2.9.3.4. --warning-as-error (-Werror)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-warning-as-error)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-warning-as-error "Permalink to this headline")

Make all warnings into errors.

##### [4.2.9.3.5. --suppress-arch-warning (-suppress-arch-warning)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#suppress-arch-warning-suppress-arch-warning)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#suppress-arch-warning-suppress-arch-warning "Permalink to this headline")

Suppress the warning that otherwise is printed when object does not contain code for target arch.

##### [4.2.9.3.6. --suppress-stack-size-warning (-suppress-stack-size-warning)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#suppress-stack-size-warning-suppress-stack-size-warning)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#suppress-stack-size-warning-suppress-stack-size-warning "Permalink to this headline")

Suppress the warning that otherwise is printed when stack size cannot be determined.

##### [4.2.9.3.7. --dump-callgraph (-dump-callgraph)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dump-callgraph-dump-callgraph)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dump-callgraph-dump-callgraph "Permalink to this headline")

Dump information about the callgraph and register usage.

##### [4.2.9.3.8. --dump-callgraph-no-demangle (-dump-callgraph-no-demangle)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#dump-callgraph-no-demangle-dump-callgraph-no-demangle)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#dump-callgraph-no-demangle-dump-callgraph-no-demangle "Permalink to this headline")

Dump callgraph information without demangling.

##### [4.2.9.3.9. --Xptxas (-Xptxas)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#xptxas-xptxas)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#xptxas-xptxas "Permalink to this headline")

Ptxas options (only used with LTO).

##### [4.2.9.3.10. --cpu-arch (-cpu-arch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cpu-arch-cpu-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cpu-arch-cpu-arch "Permalink to this headline")

Specify the name of the cpu target architecture.

##### [4.2.9.3.11. --extra-warnings (-extrawarn)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#extra-warnings-extrawarn)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#extra-warnings-extrawarn "Permalink to this headline")

Emit extra warnings about possible problems.

##### [4.2.9.3.12. --gen-host-linker-script (-ghls)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gen-host-linker-script-ghls)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gen-host-linker-script-ghls "Permalink to this headline")

Specify the type of host linker script to be generated.

##### [4.2.9.3.13. --ignore-host-info (-ignore-host-info)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#ignore-host-info-ignore-host-info)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ignore-host-info-ignore-host-info "Permalink to this headline")

Ignore information about host references, so don’t remove device code that could potentially be referenced by host.

##### [4.2.9.3.14. --keep-system-libraries (-keep-system-libraries)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#keep-system-libraries-keep-system-libraries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keep-system-libraries-keep-system-libraries "Permalink to this headline")

Don’t optimize away system library (e.g. cudadevrt) code.

##### [4.2.9.3.15. --kernels-used (-kernels-used)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#kernels-used-kernels-used)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#kernels-used-kernels-used "Permalink to this headline")

Specify kernels that are used. Can be part of a kernel name so any kernels with that string in name are matched. If this option is used, then any other kernels are considered dead-code and removed.

##### [4.2.9.3.16. --options-file (-optf)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#options-file-optf)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-file-optf "Permalink to this headline")

Include command line options from the specified file.

##### [4.2.9.3.17. --report-arch (-report-arch)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#report-arch-report-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#report-arch-report-arch "Permalink to this headline")

Report SM target arch in error messages.

##### [4.2.9.3.18. --suppress-debug-info (-suppress-debug-info)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#suppress-debug-info-suppress-debug-info)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#suppress-debug-info-suppress-debug-info "Permalink to this headline")

Do not preserve debug symbols in output. This option is ignored if used without –debug option.

##### [4.2.9.3.19. --variables-used (-variables used)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#variables-used-variables-used)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#variables-used-variables-used "Permalink to this headline")

Specify variables that are used. Can be part of a variable name so any variable with that string in name are matched. If this option is used, then any other variables are considered dead-code and potentially removed unless have other accesses from device code.

##### [4.2.9.3.20. --device-stack-protector {true|false} (-device-stack-protector)](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvlink-options-device-stack-protector)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvlink-options-device-stack-protector "Permalink to this headline")

Enable or disable the generation of stack canaries in device code (only used with LTO).

Stack canaries make it more difficult to exploit certain types of memory safety bugs involving stack-local variables.
The compiler uses heuristics to assess the risk of such a bug in each function.  Only those functions which are deemed high-risk make use of a stack canary.

**Allowed Values**

- `true`
- `false`

**Default**

The generation of stack canaries in device code is disabled.

## [4.3. NVCC Environment Variables](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-environment-variables)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-environment-variables "Permalink to this headline")

**NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS:**

The `nvcc` command line flags can be augmented using the following environment variables, if set:

`NVCC_PREPEND_FLAGS`

> Flags to be injected before the normal nvcc command line.

`NVCC_APPEND_FLAGS`

> Flags to be injected after the normal nvcc command line.

For example, after setting:

```text
export NVCC_PREPEND_FLAGS='-G -keep -arch=sm_90'

export NVCC_APPEND_FLAGS='-DNAME=" foo "'
```

The following invocation:

```text
nvcc foo.cu -o foo
```

Becomes equivalent to:

```text
nvcc -G -keep -arch=sm_90 foo.cu -o foo -DNAME=" foo "
```

These environment variables can be useful for injecting `nvcc` flags globally without modifying build scripts.

The additional flags coming from either NVCC_PREPEND_FLAGS or NVCC_APPEND_FLAGS will be listed in the verbose log (`--verbose`).

**NVCC_CCBIN:**

A default host compiler can be set using the environment variable `NVCC_CCBIN`. For example, after setting:

```text
export NVCC_CCBIN='gcc'
```

`nvcc` will choose `gcc` as the host compiler if `--compiler-bindir` is not set.

`NVCC_CCBIN` can be useful for controlling the default host compiler globally. If `NVCC_CCBIN` and `--compiler-bindir` are both set, `nvcc` will choose the host compiler specified by `--compiler-bindir`. For example:

```text
export NVCC_CCBIN='gcc'

nvcc foo.cu -ccbin='clang' -o foo
```

In this case, `nvcc` will choose `clang` as the host compiler.
