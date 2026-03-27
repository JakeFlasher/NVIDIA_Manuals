---
title: "Blackwell Warp-specialized Persistent Kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#blackwell-warp-specialized-persistent-kernel"
---

## [Blackwell Warp-specialized Persistent Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-warp-specialized-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-warp-specialized-persistent-kernel "Permalink to this headline")

Now, let’s take a look at how CLC feature is used in our [Blackwell dense GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp).

This particular warp-specialized kernel has the following warp assignment:

| Warp Role | Warp |
| --- | --- |
| MMA | 0 |
| Scheduler | 1 |
| Mainloop Load | 2 |
| Epilogue Load | 3 |
| Epilogue | 4, 5, 6, 7 |

Scheduler warp is the producer of the CLC pipeline. The consumers are the MMA, Mainloop Load, Epilogue Load and Epilogue warps. In addition, the scheduler warp is its own consumer! This is because it needs the `success` information from the query to terminate the persistent loop on end-of-grid.

The CLC pipeline has a depth of 3 to overlap the CLC operations of multiple waves for latency hiding. The first `ClcID` is the preloaded `blockIdx`, which does not require CLC query and is fully static.
