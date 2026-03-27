---
title: "Cluster Launch Control Pipeline Class"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#cluster-launch-control-pipeline-class"
---

### [Cluster Launch Control Pipeline Class](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cluster-launch-control-pipeline-class)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cluster-launch-control-pipeline-class "Permalink to this headline")

Please refer to the `PipelineCLCFetchAsync` pipeline class defined in [Cluster launch control pipeline class](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/sm100_pipeline.hpp). Cluster launch control queries can be pipelined and managed by an asynchronous pipeline with producer-consumer relationship (See
[pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) document). The producer is the scheduler warp of the 0th CTA in the cluster and the consumers are all warps that need `ClcID`s.

To setup a CLC pipeline correctly, we need to make sure the params are set to the right values:

- `transaction_bytes` is `16` as CLC will return a 16B response and store it in the specified shared memory address.
- `consumer_arv_count` is the thread count of all the consumer warps in the cluster.
- `producer_arv_count` is `1` because only one thread from scheduler warp will be elected to issue `clusterlaunchcontrol.try_cancel`.
- `producer_blockid` is `0` to denote that the first CTA in the cluster is producing.
