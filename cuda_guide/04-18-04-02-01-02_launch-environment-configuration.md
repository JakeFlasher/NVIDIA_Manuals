---
title: "4.18.4.2.1.2. Launch Environment Configuration"
section: "4.18.4.2.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#launch-environment-configuration"
---

##### [4.18.4.2.1.2. Launch Environment Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#launch-environment-configuration)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#launch-environment-configuration "Permalink to this headline")

All global device configuration settings (for example, shared memory and L1 cache size as returned from `cudaDeviceGetCacheConfig()`, and device limits returned from `cudaDeviceGetLimit()`) will be inherited from the parent. Likewise, device limits such as stack size will remain as-configured.

For host-launched kernels, per-kernel configurations set from the host will take precedence over the global setting. These configurations will be used when the kernel is launched from the device as well. It is not possible to reconfigure a kernel’s environment from the device.
