---
title: "Static Persistent Kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#static-persistent-kernel"
---

#### [Static Persistent Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#static-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#static-persistent-kernel "Permalink to this headline")

```c++
// Static Persistent Kernel
__device__ static_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, workCoordinates) = staticTileScheduler.fetch_next_work();
  } while (isValidId);
}
```
