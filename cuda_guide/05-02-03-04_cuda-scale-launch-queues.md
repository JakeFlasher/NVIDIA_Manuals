---
title: "5.2.3.4. CUDA_SCALE_LAUNCH_QUEUES"
section: "5.2.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues"
---

### [5.2.3.4. CUDA_SCALE_LAUNCH_QUEUES](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-scale-launch-queues)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-scale-launch-queues "Permalink to this headline")

The environment variable specifies the scaling factor for the size of the queues available for launching work (command buffer), namely the total number of pending kernels or host/device copy operations that can be enqueued on a device.

**Possible Values**: `0.25x`, `0.5x`, `2x`, `4x`

- Any value other than `0.25x`, `0.5x`, `2x` or `4x` is interpreted as `1x`.

**Example**:

```bash
CUDA_SCALE_LAUNCH_QUEUES=2x
```
