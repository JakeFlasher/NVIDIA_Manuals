---
title: "4.16.4. Multicast Memory Sharing"
section: "4.16.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing"
---

## [4.16.4. Multicast Memory Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#multicast-memory-sharing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#multicast-memory-sharing "Permalink to this headline")

The [Multicast Object Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST/)
provide a way for the application to create multicast objects and, in combination with the [Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
described above, allow applications to leverage NVLink SHARP on supported NVLink connected GPUs connected with NVSwitch. NVLink SHARP
allows CUDA applications to leverage in-fabric computing to accelerate operations like broadcast and reductions between GPUs connected with
NVSwitch. For this to work, multiple NVLink connected GPUs form a multicast team and each GPU from the team backs up a multicast object with
physical memory. So a multicast team of N GPUs has N physical replicas of a multicast object, each local to one participating GPU.
The [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)
using mappings of multicast objects work with all replicas of the multicast object.

To work with multicast objects, an application needs to

- Query multicast support
- Create a multicast handle with `cuMulticastCreate`.
- Share the multicast handle with all processes that control a GPU which should participate in a multicast team. This works with `cuMemExportToShareableHandle` as described above.
- Add all GPUs that should participate in the multicast team with `cuMulticastAddDevice`.
- For each participating GPU, bind physical memory allocated with `cuMemCreate` as described above to the multicast handle. All devices need to be added to the multicast team before binding memory on any device.
- Reserve an address range, map the multicast handle and set access rights as described above for regular unicast mappings. Unicast and multicast mappings to the same physical memory are possible. See the [Virtual Aliasing Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#virtual-aliasing-support) section above on how to ensure consistency between multiple mappings to the same physical memory.
- Use the [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/) with the multicast mappings.

The `multi_node_p2p` example in the [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models/) GitHub
repository contains a complete example using fabric memory including multicast objects to leverage NVLink SHARP. Please note that this example is
for developers of libraries like NCCL or NVSHMEM. It shows how higher-level programming models like NVSHMEM work internally within a (multi-node)
NVLink domain. Application developers generally should use the higher-level MPI, NCCL, or NVSHMEM interfaces instead of this API.
