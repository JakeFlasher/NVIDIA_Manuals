---
title: "5. Compilation Options"
section: "5"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-options"
---

# [5. Compilation Options](https://docs.nvidia.com/cuda/ptx-compiler-api#compilation-options)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-options "Permalink to this headline")

This chapter describes options supported by `nvPTXCompilerCompile()` API.

Option names with two preceding dashes (`--`) are long option names and option names with one
preceding dash (`-`) are short option names. Short option names can be used instead of long option
names. When a compile option takes an argument, an assignment operator (`=`) is used to separate
the compile option argument from the compile option name, e.g., `"--gpu-name=sm_80"`.
Alternatively, the compile option name and the argument can be specified in
separate strings without an assignment operator, .e.g, `"--gpu-name"``"sm_80"`.

`--allow-expensive-optimizations` (`-allow-expensive-optimizations`)

> _Enable (disable) to allow compiler to perform expensive optimizations using maximum available
> resources (memory and compile-time)._
>
>
> If unspecified, default behavior is to enable this feature for optimization level >= `O2`.

`--compile-as-tools-patch` (`-astoolspatch`)

> _Compile patch code for CUDA tools._
>
>
> Shall not be used in conjunction with `-c` or `-ewp`.
>
>
> Some PTX ISA features may not be usable in this compilation mode.

`--compile-only` (`-c`)

> _Generate relocatable object._

`--def-load-cache` (`-dlcm`)

> _Default cache modifier on global/generic load._

`--def-store-cache` (`-dscm`)

> _Default cache modifier on global/generic store._

`--device-debug` (`-g`)

> _Generate debug information for device code._

`--device-function-maxrregcount N` (`-func-maxrregcount`)

> _When compiling with -c option, specify the maximum number of registers that device functions can use._
>
>
> This option is ignored for whole-program compilation and does not affect registers used by entry
> functions. For device functions, this option overrides the value specified by `--maxrregcount`
> option. If neither `--device-function-maxrregcount` nor `--maxrregcount` is specified, then no
> maximum is assumed.
>
>
> > **Note**
> >
> > Under certain situations, `static` device functions can safely inherit a higher register count
> > from the caller entry function. In such cases, ptx compiler may apply the higher count for
> > compiling the static function.
>
>
> Value less than the minimum registers required by ABI will be bumped up by the compiler to ABI
> minimum limit.

`--disable-optimizer-constants` (`-disable-optimizer-consts`)

> _Disable use of optimizer constant bank._

`--disable-warnings` (`-w`)

> _Inhibit all warning messages._

`--dont-merge-basicblocks` (`-no-bb-merge`)

> _Prevents basic block merging, at a slight perfomance cost._
>
>
> Normally ptx compiler attempts to merge consecutive basic blocks as part of its optimization
> process. However, for debuggable code this is very confusing. This option prevents merging
> consecutive basic blocks.

`--entry entry,...` (`-e`)

> _Specify the entry functions for which code must be generated._
>
>
> Entry function names for this option must be specified in the mangled name.

`--extensible-whole-program` (`-ewp`)

> _Generate extensible whole program device code, which allows some calls to not be resolved until
> linking with libcudadevrt._

`--fmad` (`-fmad`)

> _Enables (disables) the contraction of floating-point multiplies and adds/subtracts into
> floating-point multiply-add operations (FMAD, FFMA, or DFMA)_
>
>
> Default value: `true`

`--force-load-cache` (`-flcm`)

> _Force specified cache modifier on global/generic load._

`--force-store-cache` (`-fscm`)

> _Force specified cache modifier on global/generic store._

`--generate-line-info` (`-lineinfo`)

> _Generate line-number information for device code._

`--gpu-name gpuname` (`-arch`)

> _Specify name of NVIDIA GPU to generate code for._
>
>
> This option also takes virtual compute architectures, in which case code generation is
> suppressed. This can be used for parsing only.
>
>
> PTX for .target sm_XY can be compiled to all GPU targets sm_MN, sm_MNa, sm_MNf where MN >= XY.
> PTX for .target sm_XYf can be compiled to GPU targets sm_XZ, sm_XZf, sm_XZa where Z >= Y and
> sm_XY and sm_XZ belong in same family. PTX with .target sm_XYa can only be compiled to GPU
> target sm_XYa.
>
>
> **Allowed Values**
>
>
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
>
>
> Default value: `sm_75`

`--maxrregcount N` (`-maxrregcount`)

> _Specify the maximum amount of registers that GPU functions can use._
>
>
> Until a function-specific limit, a higher value will generally increase the performance of
> individual GPU threads that execute this function. However, because thread registers are allocated
> from a global register pool on each GPU, a higher value of this option will also reduce the
> maximum thread block size, thereby reducing the amount of thread parallelism. Hence, a good
> maxrregcount value is the result of a trade-off.
>
>
> If this option is not specified, then no maximum is assumed. Value less than the minimum registers
> required by ABI will be bumped up by the compiler to ABI minimum limit. User program may not be
> able to make use of all registers as some registers are reserved by compiler.

`--opt-level N` (`-O`)

> _Specify optimization level._
>
>
> Default value: `3`.

`--position-independent-code` (`-pic`)

> _Generate position-independent code._
>
>
> Default value:
>
>
> For whole-program compilation: `true`.
>
>
> Otherwise: `false`.

`--preserve-relocs` (`-preserve-relocs`)

> _This option will make ptx compiler to generate relocatable references for variables and preserve
> relocations generated for them in linked executable._

`--return-at-end` (`-ret-end`)

> _Prevents optimizing return instruction at end of program_
>
>
> Normally ptx compiler optimizes return at the end of program. However, for debuggable code this
> causes problems setting breakpoint at the end. This option prevents ptxas from optimizing this
> last return instruction.

`--sanitize` (`-sanitize`)

> _Generate instrumented code with specified compute-sanitizer tool._
>
>
> Currently, only the value `memcheck` (memory check) is supported. Binary needs to be
> executed using `compute-sanitizer` at runtime, otherwise behavior is undefined.
> For more informations about compile time instrumentation and the compute sanitizer,
> consult the [corresponding documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer#compile-time-patching).

`--suppress-async-bulk-multicast-advisory-warning` (`-suppress-async-bulk-multicast-advisory-warning`)

> _Suppress the warning on use of .multicast::cluster modifier on cp.async.bulk{.tensor} instruction with sm_90._

`--suppress-stack-size-warning` (`-suppress-stack-size-warning`)

> _Suppress the warning that otherwise is printed when stack size cannot be determined._

`--verbose` (`-v`)

> _Enable verbose mode which prints code generation statistics._

`--warn-on-double-precision-use` (`-warn-double-usage`)

> _Warning if double(s) are used in an instruction._

`--warn-on-local-memory-usage` (`-warn-lmem-usage`)

> _Warning if local memory is used._

`--warn-on-spills` (`-warn-spills`)

> _Warning if registers are spilled to local memory._

`--warning-as-error` (`-Werror`)

> _Make all warnings into errors._

`--maxntid` (`-maxntid`)

> _Specify the maximum number of threads that a thread block can have._
>
>
> This option will be ignored if used along with `-maxrregcount` option. This option is also ignored
> for entry functions that have `.maxntid` directive specified.

`--minnctapersm` (`-minnctapersm`)

> _Specify the minimum number of CTAs to be mapped to an SM._
>
>
> This option will be ignored if used along with `-maxrregcount` option. This option is also ignored
> for entry functions that have `.minnctapersm` directive specified.

`--override-directive-values` (`-override-directive-values`)

> _Override the PTX directives values by the corresponding option values._
>
>
> This option is effective only for `-minnctapersm`, `-maxntid` and `-maxregcount` options.

`--make-errors-visible-at-exit` (`-make-errors-visible-at-exit`)

> _Generate required instructions at exit point to make memory faults and errors visible at exit._

`--oFast-compile` (`-Ofc`)

> _Specify level to prefer device code compilation speed._
>
>
> Default value: `0`.

`--device-stack-protector` (`-device-stack-protector`)

> _Enable or disable the generation of stack canaries in device code._
>
>
> Stack canaries make it more difficult to exploit certain types of memory safety bugs involving
> stack-local variables. The compiler uses heuristics to assess the risk of such a bug in each function.
> Only those functions which are deemed high-risk make use of a stack canary.

`--g-tensor-memory-access-check` (`-g-tmem-access-check`)

> _Enable tensor memory access checks for tcgen05 operations._
>
>
> This option is enabled by default with `--device-debug` (`-g`) option.

`--gno-tensor-memory-access-check` (`-gno-tmem-access-check`)

> _Disable tensor memory access checks for tcgen05 operations._
>
>
> This option will override the `--g-tensor-memory-access-check` option if both are specified.

`--split-compile` (`-split-compile`)

> _Specify the maximum amount of concurrent threads to be utilized when running compiler optimizations._
>
>
> If value specified is `1`, option will be ignored.
> If value specified is `0`, then the number of threads will be the number of CPUs on
> the underlying machine.
