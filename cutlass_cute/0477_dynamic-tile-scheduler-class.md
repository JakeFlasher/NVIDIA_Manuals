---
title: "Dynamic tile scheduler class"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#dynamic-tile-scheduler-class"
---

### [Dynamic tile scheduler class](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#dynamic-tile-scheduler-class)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#dynamic-tile-scheduler-class "Permalink to this headline")

Please refer to `PersistentTileSchedulerSm100` class defined in [sm100 dynamic persistent tile scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp).

There are two important methods of the CLC scheduler class. The first is `advance_to_next_work`, which is intended to be executed by one elected thread from the scheduler warp. It effectively sends out the CLC query to the CLC. A CLC query response will be broadcast to the same shared memory address of all CTAs in the cluster.

The other method is named `get_current_work`. It simply loads the CLC response from the shared memory buffer indexed by a pipeline state.

The CLC pipeline and scheduler classes are used together to ensure correct functionality and necessary synchronization of CLC feature. Please refer to [cluster launch control pipeline unit test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/pipeline/pipeline_cluster_launch_control_async_warp_specialized_blackwell.cu).
