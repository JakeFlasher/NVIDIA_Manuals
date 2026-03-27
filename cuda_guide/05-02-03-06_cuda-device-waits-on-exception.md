---
title: "5.2.3.6. CUDA_DEVICE_WAITS_ON_EXCEPTION"
section: "5.2.3.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-waits-on-exception"
---

### [5.2.3.6. CUDA_DEVICE_WAITS_ON_EXCEPTION](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-device-waits-on-exception)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-device-waits-on-exception "Permalink to this headline")

The environment variable controls the behavior of a CUDA application when an exception (error) occurs.

When enabled, a CUDA application will halt and wait when a device-side exception occurs, allowing a debugger, such as  [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html), to be attached to inspect the live GPU state before the process exits or continues.

**Possible Values**:

- `0`: Default behavior.
- `1`: Halt when a device exception occurs.

**Example**:

```bash
CUDA_DEVICE_WAITS_ON_EXCEPTION=1
```
