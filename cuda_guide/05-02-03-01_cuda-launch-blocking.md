---
title: "5.2.3.1. CUDA_LAUNCH_BLOCKING"
section: "5.2.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-launch-blocking"
---

### [5.2.3.1. CUDA_LAUNCH_BLOCKING](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-launch-blocking)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-launch-blocking "Permalink to this headline")

The environment variable specifies whether to disable or enable asynchronous kernel launches.

Disabling asynchronous execution results in slower execution but is useful for debugging. It forces GPU work to run synchronously from the CPU’s perspective. This allows CUDA API errors to be observed at the exact API call that triggered them, rather than later in the execution. Synchronous execution is useful for debugging purposes.

**Possible Values**:

- `1`: Disables asynchronous execution.
- `0`: Asynchronous execution (default).

**Example**:

```bash
CUDA_LAUNCH_BLOCKING=1
```
