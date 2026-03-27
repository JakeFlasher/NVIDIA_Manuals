---
title: "5.2.3.2. CUDA_DEVICE_MAX_CONNECTIONS"
section: "5.2.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections"
---

### [5.2.3.2. CUDA_DEVICE_MAX_CONNECTIONS](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-device-max-connections)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-device-max-connections "Permalink to this headline")

The environment variable controls the number of concurrent compute and copy engine connections (work queues), setting both to the specified value.
If independent GPU tasks, namely kernels or copy operations launched from different CUDA streams, map to the same work queue, a false dependency is created which can lead to GPU work serialization, since the same underlying resource(s) are used.
To reduce the probability of such false dependencies, it is recommended that the work queue count, controlled via this environment variable, be greater than or equal to the number of active CUDA streams per context.

Setting this environment variable also modifies the number of copy connections, unless they are explicitly set via the `CUDA_DEVICE_MAX_COPY_CONNECTIONS` environment variable.

**Possible Values**: `1` to `32` connections, default is `8` (assumes no MPS)

**Example**:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=16
```
