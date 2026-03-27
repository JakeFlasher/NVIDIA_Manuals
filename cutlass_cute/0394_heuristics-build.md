---
title: "Build"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html#heuristics--build"
---

### [Build](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#build)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#build "Permalink to this headline")

Build CUTLASS using CMake as normal, providing heuristics-specific options to CMake. Note that hardware details are detected automatically. For offline builds, use `-DCUTLASS_LIBRARY_HEURISTICS_GPU`.
For example, here is a minimal command for Nvidia’s Hopper Architecture (sm90):

```console
$ cmake .. \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE=<path_to_your_problem_list.json> \
    -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=<number of configurations to build per problem>
...
...

$ make cutlass_profiler -j
```

This will produce a csv testlist which provides all testcases that need be run to perform autotuning over the built configurations, including kernel runtime options. The location of this file can be changed by the CMake option `-DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE`.

CUTLASS CMake currently supports the following for heuristics:

- `CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE`: Path to the file containing a json list of GEMM problems
- `CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM`: Max number of configurations the heuristic will return for each GEMM problem. The same configuration or kernel can be suggested for multiple problems.
- `CUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS`: Limits the build to only the set of kernels instantiated by the default CUTLASS CMake build flow, composing with other options such as `CUTLASS_LIBRARY_INSTANTIATION_LEVEL`. Set this to `ON` as a workaround if the heuristic suggests kernel configurations that do not build on your platform (possible for some unsupported or experimental use cases). This option is set to `OFF` by default, which builds all of the suggested configurations.
- `CUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE`: Path to the output CSV which will contain the testcases to be used for autotuning, consumable by `cutlass_profiler`.
- `CUTLASS_LIBRARY_HEURISTICS_GPU`: The GPU to use for heuristics; for instance, `H100_SXM5`. Used for offline builds. If unset, the hardware properties will be auto-detected using the Cuda Driver APIs. See `generator.py` for valid GPU strings
