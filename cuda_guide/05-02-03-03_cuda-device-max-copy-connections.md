---
title: "5.2.3.3. CUDA_DEVICE_MAX_COPY_CONNECTIONS"
section: "5.2.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-copy-connections"
---

### [5.2.3.3. CUDA_DEVICE_MAX_COPY_CONNECTIONS](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-device-max-copy-connections)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-device-max-copy-connections "Permalink to this headline")

The environment variable controls the number of concurrent copy connections (work queues) involved in copy operations.
It affects only devices of [compute capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities) 8.0 and above.

The `CUDA_DEVICE_MAX_COPY_CONNECTIONS` overrides the value of copy connections set via `CUDA_DEVICE_MAX_CONNECTIONS`, if both were set.

**Possible Values**: `1` to `32` connections, default is `8` (assumes no MPS)

**Example**:

```bash
CUDA_DEVICE_MAX_COPY_CONNECTIONS=16
```
