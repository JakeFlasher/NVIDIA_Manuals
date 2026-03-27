---
title: "5.2.1.3. CUDA_MANAGED_FORCE_DEVICE_ALLOC"
section: "5.2.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-managed-force-device-alloc"
---

### [5.2.1.3. CUDA_MANAGED_FORCE_DEVICE_ALLOC](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-managed-force-device-alloc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-managed-force-device-alloc "Permalink to this headline")

The environment variable alters how [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory) is physically stored in multi-GPU systems.

**Possible Values**: Numerical value, either zero or non-zero.

- **Non-zero value**: Forces the driver to use device memory for physical storage. All devices used in the process that support managed memory must be peer-to-peer compatible. Otherwise, `cudaErrorInvalidDevice` is returned.
- `0`: Default behavior.

**Examples**:

```bash
CUDA_MANAGED_FORCE_DEVICE_ALLOC=0
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 # force device memory
```

---
