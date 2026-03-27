---
title: "Handling Unresponsive/Hung Kernels"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#handling-unresponsive-hung-kernels"
---

### [Handling Unresponsive/Hung Kernels](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#handling-unresponsive-hung-kernels)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#handling-unresponsive-hung-kernels "Permalink to this headline")

When a kernel becomes unresponsive and `SIGINT` (`CTRL+C`) fails to terminate it,
you can follow these steps to forcefully terminate the process:

1. Use `CTRL+Z` to suspend the unresponsive kernel
2. Execute the following command to terminate the suspended process:

```bash
# Terminate the most recently suspended process
kill -9 $(jobs -p | tail -1)
```

CuTe DSL can also be debugged using standard NVIDIA CUDA tools.
