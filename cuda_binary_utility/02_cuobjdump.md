---
title: "2. cuobjdump"
section: "2"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump"
---

# [2. cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities#cuobjdump)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump "Permalink to this headline")

`cuobjdump` extracts information from CUDA binary files (both standalone and those embedded in host binaries) and presents them in human readable format. The output of `cuobjdump` includes CUDA assembly code for each kernel, CUDA ELF section headers, string tables, relocators and other CUDA specific sections. It also extracts embedded ptx text from host binaries.

For a list of CUDA assembly instruction set of each GPU architecture, see [Instruction Set Reference](https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-ref).

## [2.1. Usage](https://docs.nvidia.com/cuda/cuda-binary-utilities#usage)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#usage "Permalink to this headline")

`cuobjdump` accepts a single input file each time it’s run. The basic usage is as following:

```text
cuobjdump [options] <file>
```

To disassemble a standalone cubin or cubins embedded in a host executable and show CUDA assembly of the kernels, use the following command:

```text
cuobjdump -sass <input file>
```

To dump cuda elf sections in human readable format from a cubin file, use the following command:

```text
cuobjdump -elf <cubin file>
```

To extract ptx text from a host binary, use the following command:

```text
cuobjdump -ptx <host binary>
```

Here’s a sample output of `cuobjdump`:

```text
$ cuobjdump -ptx -sass add.o

Fatbin elf code:
================
arch = sm_100
code version = [1,8]
host = linux
compile_size = 64bit

     code for sm_100
     .target sm_100

             Function : _Z3addPfS_S_
     .headerflags    @"EF_CUDA_SM100 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM100)"
     /*0000*/                   LDC R1, c[0x0][0x37c] ;       /* 0x0000df00ff017b82 */
                                                              /* 0x000fe20000000800 */
     /*0010*/                   S2R R9, SR_TID.X ;            /* 0x0000000000097919 */
                                                              /* 0x000e2e0000002100 */
     /*0020*/                   S2UR UR6, SR_CTAID.X ;        /* 0x00000000000679c3 */
                                                              /* 0x000e220000002500 */
     /*0030*/                   LDCU.64 UR4, c[0x0][0x358] ;  /* 0x00006b00ff0477ac */
                                                              /* 0x000e6e0008000a00 */
     /*0040*/                   LDC R0, c[0x0][0x360] ;       /* 0x0000d800ff007b82 */
                                                              /* 0x000e300000000800 */
     /*0050*/                   LDC.64 R2, c[0x0][0x380] ;    /* 0x0000e000ff027b82 */
                                                              /* 0x000eb00000000a00 */
     /*0060*/                   LDC.64 R4, c[0x0][0x388] ;    /* 0x0000e200ff047b82 */
                                                              /* 0x000ee20000000a00 */
     /*0070*/                   IMAD R9, R0, UR6, R9 ;        /* 0x0000000600097c24 */
                                                              /* 0x001fce000f8e0209 */
     /*0080*/                   LDC.64 R6, c[0x0][0x390] ;    /* 0x0000e400ff067b82 */
                                                              /* 0x000e220000000a00 */
     /*0090*/                   IMAD.WIDE R2, R9, 0x4, R2 ;   /* 0x0000000409027825 */
                                                              /* 0x004fcc00078e0202 */
     /*00a0*/                   LDG.E R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
                                                              /* 0x002ea2000c1e1900 */
     /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;   /* 0x0000000409047825 */
                                                              /* 0x008fcc00078e0204 */
     /*00c0*/                   LDG.E R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
                                                              /* 0x000ea2000c1e1900 */
     /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;   /* 0x0000000409067825 */
                                                              /* 0x001fc800078e0206 */
     /*00e0*/                   FADD R9, R2, R5 ;             /* 0x0000000502097221 */
                                                              /* 0x004fca0000000000 */
     /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;  /* 0x0000000906007986 */
                                                              /* 0x000fe2000c101904 */
     /*0100*/                   EXIT ;                        /* 0x000000000000794d */
                                                              /* 0x000fea0003800000 */
     /*0110*/                   BRA 0x110;                    /* 0xfffffffc00fc7947 */
                                                              /* 0x000fc0000383ffff */
     /*0120*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0130*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0140*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0150*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0160*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0170*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0180*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*0190*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01a0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01b0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01c0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01d0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01e0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
     /*01f0*/                   NOP;                          /* 0x0000000000007918 */
                                                              /* 0x000fc00000000000 */
             ..........
```

```text
Fatbin ptx code:
================
arch = sm_100
code version = [8,8]
host = linux
compile_size = 64bit
compressed
ptxasOptions =

//
//
//
//
//
//

.version 8.8
.target sm_100
.address_size 64

//

.visible .entry _Z3addPfS_S_(
.param .u64 .ptr .align 1 _Z3addPfS_S__param_0,
.param .u64 .ptr .align 1 _Z3addPfS_S__param_1,
.param .u64 .ptr .align 1 _Z3addPfS_S__param_2
)
{
.reg .b32 %r<5>;
.reg .f32 %f<4>;
.reg .b64 %rd<11>;

ld.param.u64 %rd1, [_Z3addPfS_S__param_0];
ld.param.u64 %rd2, [_Z3addPfS_S__param_1];
ld.param.u64 %rd3, [_Z3addPfS_S__param_2];
cvta.to.global.u64 %rd4, %rd3;
cvta.to.global.u64 %rd5, %rd2;
cvta.to.global.u64 %rd6, %rd1;
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
mul.wide.s32 %rd7, %r4, 4;
add.s64 %rd8, %rd6, %rd7;
ld.global.f32 %f1, [%rd8];
add.s64 %rd9, %rd5, %rd7;
ld.global.f32 %f2, [%rd9];
add.f32 %f3, %f1, %f2;
add.s64 %rd10, %rd4, %rd7;
st.global.f32 [%rd10], %f3;
ret;
}
```

As shown in the output, the `a.out` host binary contains cubin and ptx code for sm_100.

To list cubin files in the host binary use `-lelf` option:

```text
$ cuobjdump a.out -lelf
ELF file    1: add_new.sm_100.cubin
ELF file    2: add_new.sm_120.cubin
ELF file    3: add_old.sm_100.cubin
ELF file    4: add_old.sm_120.cubin
```

To extract all the cubins as files from the host binary use `-xelf all` option:

```text
$ cuobjdump a.out -xelf all
Extracting ELF file    1: add_new.sm_100.cubin
Extracting ELF file    2: add_new.sm_120.cubin
Extracting ELF file    3: add_old.sm_100.cubin
Extracting ELF file    4: add_old.sm_120.cubin
```

To extract the cubin named `add_new.sm_100.cubin`:

```text
$ cuobjdump a.out -xelf add_new.sm_100.cubin
Extracting ELF file    1: add_new.sm_100.cubin
```

To extract only the cubins containing `_old` in their names:

```text
$ cuobjdump a.out -xelf _old
Extracting ELF file    1: add_old.sm_100.cubin
Extracting ELF file    2: add_old.sm_120.cubin
```

You can pass any substring to `-xelf` and `-xptx` options. Only the files having the substring in the name will be extracted from the input binary.

To dump common and per function resource usage information:

```text
$ cuobjdump test.cubin -res-usage

Resource usage:
 Common:
  GLOBAL:56 CONSTANT[3]:28
 Function calculate:
  REG:24 STACK:8 SHARED:0 LOCAL:0 CONSTANT[0]:472 CONSTANT[2]:24 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function mysurf_func:
  REG:38 STACK:8 SHARED:4 LOCAL:0 CONSTANT[0]:532 TEXTURE:8 SURFACE:7 SAMPLER:0
 Function mytexsampler_func:
  REG:42 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:472 TEXTURE:4 SURFACE:0 SAMPLER:1
```

Note that value for REG, TEXTURE, SURFACE and SAMPLER denotes the count and for other resources it denotes no. of byte(s) used.

## [2.2. Command-line Options](https://docs.nvidia.com/cuda/cuda-binary-utilities#command-line-options)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#command-line-options "Permalink to this headline")

[Table 2](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump-options-table) contains supported command-line options of `cuobjdump`, along with a description of what each option does. Each option has a long name and a short name, which can be used interchangeably.

| Option (long) | Option (short) | Description |
| --- | --- | --- |
| `--all-fatbin` | `-all` | Dump all fatbin sections. By default will only dump contents of executable fatbin (if exists), else relocatable fatbin if no executable fatbin. |
| `--dump-elf` | `-elf` | Dump ELF Object sections. |
| `--dump-elf-symbols` | `-symbols` | Dump ELF symbol names. |
| `--dump-ptx` | `-ptx` | Dump PTX for all listed device functions. |
| `--dump-sass` | `-sass` | Dump CUDA assembly for a single cubin file or all cubin files embedded in the binary. |
| `--dump-resource-usage` | `-res-usage` | Dump resource usage for each ELF. Useful in getting all the resource usage information at one place. |
| `--extract-elf <partial file name>,...` | `-xelf` | Extract ELF file(s) name containing <partial file name> and save as file(s). Use `all` to extract all files. To get the list of ELF files use -lelf option. Works with host executable/object/library and external fatbin. All `dump` and `list` options are ignored with this option. |
| `--extract-ptx <partial file name>,...` | `-xptx` | Extract PTX file(s) name containing <partial file name> and save as file(s). Use `all` to extract all files. To get the list of PTX files use -lptx option. Works with host executable/object/library and external fatbin. All `dump` and `list` options are ignored with this option. |
| `--extract-text <partial file name>,...` | `-xtext` | Extract text binary encoding file(s) name containing <partial file name> and save as file(s). Use ‘all’ to extract all files. To get the list of text binary encoding use -ltext option. All ‘dump’ and ‘list’ options are ignored with this option. |
| `--function <function name>,...` | `-fun` | Specify names of device functions whose fat binary structures must be dumped. |
| `--function-index <function index>,...` | `-findex` | Specify symbol table index of the function whose fat binary structures must be dumped. |
| `--gpu-architecture <gpu architecture name>` | `-arch` | Specify GPU Architecture for which information should be dumped. Allowed values for this option: `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_90a`, `sm_100`, `sm_100a`, `sm_100f`, `sm_103`, `sm_103a`, `sm_103f`, `sm_110`, `sm_110a`, `sm_110f`, `sm_120`, `sm_120a`, `sm_120f`, `sm_121`, `sm_121a`, `sm_121f`. |
| `--help` | `-h` | Print this help information on this tool. |
| `--list-elf` | `-lelf` | List all the ELF files available in the fatbin. Works with host executable/object/library and external fatbin. All other options are ignored with this flag. This can be used to select particular ELF with -xelf option later. |
| `--list-ptx` | `-lptx` | List all the PTX files available in the fatbin. Works with host executable/object/library and external fatbin. All other options are ignored with this flag. This can be used to select particular PTX with -xptx option later. |
| `--list-text` | `-ltext` | List all the text binary function names available in the fatbin. All other options are ignored with the flag. This can be used to select particular function with -xtext option later. |
| `--options-file <file>,...` | `-optf` | Include command line options from specified file. |
| `--sort-functions` | `-sort` | Sort functions when dumping sass. |
| `--version` | `-V` | Print version information on this tool. |
