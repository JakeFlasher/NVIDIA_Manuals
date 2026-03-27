---
title: "Developer Tools"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/quickstart.html#developer-tools"
---

## [Developer Tools](https://docs.nvidia.com/cuda/cutile-python#developer-tools)[](https://docs.nvidia.com/cuda/cutile-python/#developer-tools "Permalink to this headline")

[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) can profile cuTile Python kernels in the same way as SIMT CUDA kernels. With NVIDIA Nsight Compute installed, the quickstart vector addition kernel introduced here can be profiled using the following command to create a profile:

```bash
ncu -o VecAddProfile --set detailed python3 VectorAdd_quickstart.py
```

This profile can then be loaded in a graphical instance of Nsight Compute and the kernel `vector_add` selected to see statistics about the kernel.

> **Note**
>
> Capturing detailed statistics for cuTile Python kernels requires running on NVIDIA Driver equals or later than r580.126.09 (linux) or r582.16 (windows).
