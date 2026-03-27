---
title: "Profile"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html#profile"
---

### [Profile](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#profile)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#profile "Permalink to this headline")

Use the emitted testlist CSV with `cutlass_profiler` to collect performance data, which can be used to determine the fastest built kernel configuration for each of the input problems. Example which profiles each testcase for a fixed 50ms:

```console
cutlass_profiler --operation=Gemm --testlist-file=<path_to_your_testlist.csv> --profiling-iterations=0 --profiling-duration=50 --verification-enabled=false --output=<path_to_outfile>
```
