---
title: "CUTLASS Profiler"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#code_organization--cutlass-profiler"
---

### [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-profiler)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-profiler "Permalink to this headline")

The CUTLASS Profiler is designed to load the CUTLASS Instance Library and execute all operations contained therein.
This command-line driven application constructs an execution environment for evaluating functionality and performance.
It is implemented in

```console
tools/
  profiler/
```

and may be built as follows.

```console
$ make cutlass_profiler -j
```

[Further details about the CUTLASS Profiler are described here.](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
