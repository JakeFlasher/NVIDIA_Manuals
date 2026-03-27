---
title: "5.2.3.7. CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT"
section: "5.2.3.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-default-persisting-l2-cache-percentage-limit"
---

### [5.2.3.7. CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-device-default-persisting-l2-cache-percentage-limit)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-device-default-persisting-l2-cache-percentage-limit "Permalink to this headline")

The environment variable controls the default “set-aside” portion of the GPU’s L2 cache reserved for [persisting accesses](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#l2-set-aside), expressed as a percentage of the L2 size.

It is relevant for GPUs that support persistent L2 cache, specifically devices with [compute capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities) 8.0 or higher when using the [CUDA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html). The environment variable must be set before starting the CUDA MPS Control Daemon, namely before running the `nvidia-cuda-mps-control -d` command.

**Possible Values**: Percentage value between 0 and 100, default is 0.

**Example**:

```bash
CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=25 # 25%
```
