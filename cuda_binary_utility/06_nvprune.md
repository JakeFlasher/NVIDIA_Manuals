---
title: "6. nvprune"
section: "6"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune"
---

# [6. nvprune](https://docs.nvidia.com/cuda/cuda-binary-utilities#nvprune)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune "Permalink to this headline")

`nvprune` prunes host object files and libraries to only contain device code for the specified targets.

## [6.1. Usage](https://docs.nvidia.com/cuda/cuda-binary-utilities#id9)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#id9 "Permalink to this headline")

`nvprune` accepts a single input file each time it’s run, emitting a new output file. The basic usage is as following:

```text
nvprune [options] -o <outfile> <infile>
```

The input file must be either a relocatable host object or static library (not a host executable), and the output file will be the same format.

Either the –arch or –generate-code option must be used to specify the target(s) to keep. All other device code is discarded from the file. The targets can be either a sm_NN arch (cubin) or compute_NN arch (ptx).

For example, the following will prune libcublas_static.a to only contain sm_120 cubin rather than all the targets which normally exist:

```text
nvprune -arch sm_120 libcublas_static.a -o libcublas_static120.a
```

Note that this means that libcublas_static120.a will not run on any other architecture, so should only be used when you are building for a single architecture.

## [6.2. Command-line Options](https://docs.nvidia.com/cuda/cuda-binary-utilities#nvprune-options)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune-options "Permalink to this headline")

[Table 10](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune-options-table) contains supported command-line options of `nvprune`, along with a description of what each option does. Each option has a long name and a short name, which can be used interchangeably.

| Option (long) | Option (short) | Description |
| --- | --- | --- |
| `--arch <gpu architecture name>,...` | `-arch` | Specify the name of the NVIDIA GPU architecture which will remain in the object or library. |
| `--generate-code` | `-gencode` | This option is same format as nvcc –generate-code option, and provides a way to specify multiple architectures which should remain in the object or library. Only the ‘code’ values are used as targets to match. Allowed keywords for this option: ‘arch’,’code’. |
| `--no-relocatable-elf` | `-no-relocatable-elf` | Don’t keep any relocatable ELF. |
| `--output-file` | `-o` | Specify name and location of the output file. |
| `--help` | `-h` | Print this help information on this tool. |
| `--options-file <file>,...` | `-optf` | Include command line options from specified file. |
| `--version` | `-V` | Print version information on this tool. |
