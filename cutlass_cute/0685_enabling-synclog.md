---
title: "Enabling synclog"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#enabling-synclog"
---

### [Enabling synclog](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#enabling-synclog)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#enabling-synclog "Permalink to this headline")

To enable `synclog`, add the -DCUTLASS_ENABLE_SYNCLOG=1 flag during compilation. From the CUTLASS root directory:

```console
$ mkdir build && cd build &&
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_SYNCLOG=1
```
