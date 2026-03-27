---
title: "5.6.4.7. API Errors and Launch Failures"
section: "5.6.4.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#api-errors-and-launch-failures"
---

### [5.6.4.7. API Errors and Launch Failures](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#api-errors-and-launch-failures)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#api-errors-and-launch-failures "Permalink to this headline")

As usual for the CUDA runtime, any function may return an error code. The last error code returned is recorded and may be retrieved via the `cudaGetLastError()` call. Errors are recorded per-thread, so that each thread can identify the most recent error that it has generated. The error code is of type `cudaError_t`.

Similar to a host-side launch, device-side launches may fail for many reasons (invalid arguments, etc). The user must call `cudaGetLastError()` to determine if a launch generated an error, however lack of an error after launch does not imply the child kernel completed successfully.

For device-side exceptions, e.g., access to an invalid address, an error in a child grid will be returned to the host.
