---
title: "4.20.3.3. Retrieve Per-thread Default Stream Versions"
section: "4.20.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#retrieve-per-thread-default-stream-versions"
---

### [4.20.3.3. Retrieve Per-thread Default Stream Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#retrieve-per-thread-default-stream-versions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#retrieve-per-thread-default-stream-versions "Permalink to this headline")

Some CUDA driver APIs can be configured to have _default stream_ or _per-thread default stream_ semantics. Driver APIs having _per-thread default stream_ semantics are suffixed with __ptsz_ or __ptds_ in their name. For example, `cuLaunchKernel` has a _per-thread default stream_ variant named `cuLaunchKernel_ptsz`. With the Driver Entry Point Access APIs, users can request for the _per-thread default stream_ version of the driver API `cuLaunchKernel` instead of the _default stream_ version. Configuring the CUDA driver APIs for _default stream_ or _per-thread default stream_ semantics affects the synchronization behavior. More details can be found [here](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream).

The _default stream_ or _per-thread default stream_ versions of a driver API can be obtained by one of the following ways:

- Use the compilation flag `--default-stream per-thread` or define the macro `CUDA_API_PER_THREAD_DEFAULT_STREAM` to get _per-thread default stream_ behavior.
- Force _default stream_ or _per-thread default stream_ behavior using the flags `CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream` or `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream` respectively.
