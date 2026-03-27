---
title: "5.2.1.2. CUDA_DEVICE_ORDER"
section: "5.2.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-order"
---

### [5.2.1.2. CUDA_DEVICE_ORDER](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-device-order)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-device-order "Permalink to this headline")

The environment variable controls the order in which CUDA enumerates the available devices.

**Possible Values**:

- `FASTEST_FIRST`: The available devices are enumerated from fastest to slowest using a simple heuristic (default).
- `PCI_BUS_ID`: The available devices are enumerated by PCI bus ID in ascending order. The PCI bus IDs can be obtained with `nvidia-smi --query-gpu=name,pci.bus_id`.

**Examples**:

```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST
CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,pci.bus_id # Get list of PCI bus IDs
```
