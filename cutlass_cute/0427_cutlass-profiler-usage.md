---
title: "CUTLASS Profiler usage"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#cutlass-profiler-usage"
---

## [CUTLASS Profiler usage](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-profiler-usage)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-profiler-usage "Permalink to this headline")

The CUTLASS Profiler usage statement may be obtained by executing `cutlass_profiler --help` and appears as follows.

```bash
CUTLASS Performance Tool
usage:

    cutlass_profiler [options]

  --help

  --mode=<string>                                  Cutlass profiler execution mode.
                                                    --mode=profile    regular verification and profiling (default)
                                                    --mode=dry_run    no kernels are launched or workspaces allocated
                                                    --mode=enumerate  lists all operation kind and operations
                                                    --mode=trace      executes a single device-side computation with
                                                                       no other kernel launches

  --device-info                                    Prints information on all GPUs present in the system

  --operation=<operation_kind>                     CUTLASS operation to profile.

  --kernels=<string_list>                          Filter operations by kernel names. For example, call all kernels with
                                                   ("s1688" and "nt") or ("s844" and "tn" and "align8") in their
                                                   operation name using --kernels="s1688*nt, s884*tn*align8"

  --kernels-file=<path>                            Same behavior as `kernels`, but kernel names are specified in a file with
                                                   one kernel name on each line. Set of profiled kernels is the union of kernels
                                                   specified here and those specified in `kernels`.

  --ignore-kernels=<string_list>                   Excludes kernels whose names match anything in this list.

Device:
  --device=<int>                                   CUDA Device ID

  --compute-capability=<int>                       Override the compute capability.

  --llc-capacity=<capacity in KiB>                 Capacity of last-level cache in kilobytes. If this is non-zero,
                                                   profiling phases cycle through different input tensors to induce
                                                   capacity misses in the L2.

  --allocations=<name>:<device>,<name>:<device>    Pairs of allocation names to devices. If <device> is negative,
                                                   the execution device is used

Initialization:
  --initialization=<bool>                          Enables initialization (default: true). If false, device memory is
                                                   not initialized after allocation.

  --initialization-provider=<provider>             Selects initialization provider {host, device*}. (default: '*')

  --dist=<distribution>                            Data distribution of input tensors {uniform*, gaussian, identity, sequential}
                                                    --dist=uniform,min:<double>,max:<double>,scale:<integer>
                                                    --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>
                                                    --dist=sequential,start:<double>,delta:<double>,scale:<integer>
                                                    --dist=identity

  --seed=<int>                                     Random number generator seed. Used to enforce deterministic
                                                   initialization.

Library:
  --library-algo-mode=<mode>                       Indicates algorithm mode used to call libraries such as cuBLAS and cuDNN.
                                                   mode={default*,matching,best}

  --library-algos=<range-list>                     If --algorithm-mode=best, permits specifying a selection of algorithms.

Profiling:
  --workspace-count=<workspace count>              Number of discrete workspaces maintained to avoid cache-resident
                                                 If zero (default), the amount is chosen for each workload based on
                                                 capacity of the last-level cache.

  --profiling-iterations=<iterations>              Number of iterations to profile each kernel. If zero, kernels
                                                   are launched up to the profiling duration. If non-zero, this
                                                   overrides `profiling-duration` and `min-iterations`.

  --profiling-duration=<duration>                  Time to spend profiling each kernel (ms). Overriden by
                                                   `profiling-iterations` when `profiling-iterations` != 0.
                                                   Note that `min-iterations` must also be satisfied.

  --min-iterations=<iterations>                    Minimum number of iterations to spend profiling each kernel, even if
                                                   `profiling-duration` has been met.

  --warmup-iterations=<iterations>                 Number of iterations to execute each kernel prior to profiling (default: 10).

  --use-cuda-graphs=<bool>                         If true, kernels are launched in a CUDA graph. Useful when the kernel launch time is a bottleneck.

  --sleep-duration=<duration>                      Number of ms to sleep between profiling periods (ms).

  --profiling-enabled=<bool>                       If true, profiling is actually conducted.

Verification:
  --verification-enabled=<bool>                    Whether to perform verification checks.

  --epsilon=<error>                                Error threshold. Setting to zero (default) requires
                                                   bit-level equivalence.

  --nonzero-floor=<floor>                          Results whose absolute value is less than this quantity
                                                   are treated as zero for comparisons.

  --save-workspace=<string>                        Specifies when to save the GEMM inputs and results to the filesystem.
                                                    --save-workspace=never      never save workspace (default)
                                                    --save-workspace=incorrect  save workspace for incorrect results
                                                    --save-workspace=always     always save workspace

  --verification-providers=<providers>             List of providers used to verify result. (default: '*')
                                                   Gemm verification-providers {cublas*}
                                                   Conv2d verification-providers {cudnn*, device*, host}

Report:
  --append=<bool>                                  If true, result is appended to possibly existing file. Otherwise,
                                                   any existing file is overwritten.

  --output=<path>                                  Path to output file for machine readable results. Operation kind and '.csv' is appended.

  --junit-output=<path>                            Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.

  --report-not-run=<bool>                          If true, reports the status of all kernels including those that
                                                   do not satisfy the given arguments.

  --tags=<column:tag,...>                          Inserts leading columns in output table and uniform values for each
                                                   column. Useful for generating pivot tables.

  --verbose=<bool>                                 Prints human-readable text to stdout. If false, nothing is written to stdout.

About:
  --version                                        CUTLASS 2.4.0 built on Nov 19 2020 at 11:59:00

Operations:

     gemm                                          General matrix-matrix product. D = alpha * A*B + beta * C
     spgemm                                        Structured sparse GEMM. D = alpha * A*B + beta * C
     conv2d                                        Conv2d operation. Output(Tensor4D) = alpha * Input(Tensor4D) * Filter(Tensor4D) + beta * Input(Tensor4D)
     conv3d                                        Conv3d operation. Output(Tensor5D) = alpha * Input(Tensor5D) * Filter(Tensor5D) + beta * Input(Tensor5D)

For details about a particular function, specify the function name with --help.

Example:

  $ cutlass_profiler --operation=Gemm --help

  $ cutlass_profiler --operation=Conv3d --help

  $ cutlass_profiler --operation=Conv2d --help
```
