---
title: "5.2.5.1. CUDA_LOG_FILE"
section: "5.2.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#environment-variables--cuda-log-file"
---

### [5.2.5.1. CUDA_LOG_FILE](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-log-file)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-log-file "Permalink to this headline")

The environment variable specifies a location where descriptive error log messages will be printed as they occur
for supported CUDA API calls that returned an error.

For example, if one attempts to launch a kernel with an invalid grid configuration, such as `kernel<<<1, dim3(1,1,128)>>>(...)`, that kernel will fail to launch and `cudaGetLastError()` will return a generic `invalid configuration argument` error.

If the `CUDA_LOG_FILE` environment variable is set, the user can see the following descriptive error message in the log: `[CUDA][E] Block Dimensions (1,1,128) include one or more values that exceed the device limit of (1024,1024,64)` and easily determine that the specified z-dimension of the block was invalid. See [Error Log Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management) for more details.

**Possible Values**: `stdout`, `stderr`, or a valid file path (with appropriate access permissions)

**Examples**:

```bash
CUDA_LOG_FILE=stdout
CUDA_LOG_FILE=/tmp/dbg_cuda_log
```
