---
title: "Blackwell Dynamic Persistent Kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#blackwell-dynamic-persistent-kernel"
---

#### [Blackwell Dynamic Persistent Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-dynamic-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-dynamic-persistent-kernel "Permalink to this headline")

```c++
// Dynamic Persistent Kernel
__device__ clc_dynamic_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  dim3 newClcID;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, newClcID) = clcTileScheduler.fetch_next_work();
    workCoordinates = newClcID;
  } while (isValidId);
}
```
