---
title: "Non-persistent kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#non-persistent-kernel"
---

#### [Non-persistent kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#non-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#non-persistent-kernel "Permalink to this headline")

```c++
// Non-persistent kernel
__device__ non_persistent_kernel(...) {
  setup_common_data_structures();
  dim3 workCoordinates = blockIdx;
  coordinate_specific_compute(workCoordinates);
}
```
