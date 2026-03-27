---
title: "Hopper Warp Specialization"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization"
---

### [Hopper Warp Specialization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#hopper-warp-specialization)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#hopper-warp-specialization "Permalink to this headline")

Note: the following section on warp-specialization contains details that are specific
to the Hopper kernel design. Blackwell SM100 kernels have a substantially different warp-specialization structure,
however, the concept of separating out producer and consumer agents still applies.

Starting with Hopper, CUTLASS 3.0 incorporates the concept of [Warp Specialization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization)
as part of the kernel design. A thread block is partitioned into two sets of warps, [_producer_ warp group](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp) and [_consumer_ warp group](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp). The _producer_ warp group loads data from global memory into shared memory buffers using the new [Tensor Memory Accelerator (TMA)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

[_Producer_ warp group (DMA)](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) waits for the shared memory buffers to be signaled as [empty](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) by the _consumer_ warp group using the newly added **Async Pipeline class** ([refer](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)). Once the data is written into the shared memory, TMA is also updates the barrier associated with that stage to notify affected threads that the buffer has been [filled](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp). The [_Consumer_ warp group (MMA)](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) on the other hand waits for the _producer_ warp group to signal that the buffer is [filled](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) and then launches tensor core MMA operations. Finally, the _consumer_ warp group [releases](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) the buffers for the next set of TMA loads to happens.

**Warp-Specialized Persistent Cooperative kernel design**

Another flavor of Warp-Specialized kernel design being introduced starting with Hopper is the [_Warp-Specialized Persistent Cooperative_](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) kernel. Like the Warp-Specialized kernel, the concepts of warp groups and barrier synchronization between warp groups remain the same in the cooperative design.
The distinctive feature of the Warp-Specialized Persistent Cooperative kernel are the following :

- Persistent thread blocks launched to occupy as many SMs as mentioned in the [KernelHardwareInfo](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/kernel_hardware_info.hpp) struct. These persistent thread blocks are used to tile the output and thus (potentially) compute multiple output tiles through their lifetime. The main benefit this adds is amortization of the thread-block launch and kernel prologue overheads which are typical of all kernels.
- Presence of two _consumer_ warp groups cooperating on the same output tile by splitting the tile in half across the M dimension. This allows for larger tile sizes to be enabled - since the register pressure per _consumer_ warp group is reduced - and hence improving performance.

Since each thread block now computes multiple output tiles, the shape of the grid launch and the scheduling of tiles to the thread blocks is managed using the new [_Tile Scheduler_](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp). The _Tile Scheduler_ considers the shape of the _clusters_ as well as the available number of available SMs to compute a valid scheduling of the output tiles to launched thread blocks.

**Warp-Specialized Persistent Ping-Pong kernel design**

The third kernel design is the [_Warp-Specialized Persistent Ping-Pong_](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) kernel.
Like the Warp-Specialized Persistent Cooperative, kernel the concepts of warp groups, barrier synchronization between warp groups, and the shape of the grid launch remain the same in the persistent ping-pong design.
The distinctive feature of the Warp-Specialized Persistent Ping-Pong kernel is the following :

- The two _consumer_ warp groups are assigned a different output tile using the Tile Scheduler. This allows for _epilogue_ of one _consumer_ warp group to be overlapped with the math operations of the other _consumer_ warp group - thus maximizing tensor core utilization.
- The _producer_ warp group synchronizes using the [Ordered Sequence Barrier](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/pipeline.hpp) to fill buffers of the two _consumer_ warp groups one after the other in order.
