---
title: "Inter-Kernel"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/interoperability.html#inter-kernel"
---

### [Inter-Kernel](https://docs.nvidia.com/cuda/cutile-python#inter-kernel)[](https://docs.nvidia.com/cuda/cutile-python/#inter-kernel "Permalink to this headline")

Inter-kernel interoperability refers to all interoperability concerns that do not cross the kernel boundary - everything except mixing tile and SIMT code in a kernel.
This includes:

- Writing tile and SIMT kernels in the same source file.
- Linking tile and SIMT kernels into the same binary.
- Passing the same kinds of arrays to both tile and SIMT kernels.

Inter-kernel interoperability shall be supported.
