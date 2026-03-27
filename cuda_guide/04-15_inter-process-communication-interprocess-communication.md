---
title: "4.15. Interprocess Communication"
section: "4.15"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#inter-process-communication--interprocess-communication"
---

# [4.15. Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#interprocess-communication)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#interprocess-communication "Permalink to this headline")

Communication between multiple GPUs managed by different
host processes is supported through the use of
interprocess communication (IPC) APIs and IPC-shareable memory buffers,
by creating process-portable handles that are subsequently used to
obtain process-local device pointers to the device memory on peer GPUs.

Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process.
However, device pointers or event handles are not valid outside the process that created them, and therefore cannot be directly referenced by threads belonging to a different process.
To access device memory and CUDA events across processes, an application must use CUDA Interprocess Communication (IPC) or Virtual Memory Management APIs to create process-portable handles that can be shared with other processes using standard host operating system IPC mechanisms, e.g., interprocess shared memory or files.
Once the process-portable handles have been exchanged between processes, process-local device pointers must be obtained from the handles using CUDA IPC or VMM APIs.
Process-local device pointers can then be used just as they would within a single process.

The same kind of portable-handle approach used for IPC within a
single-node and single operating system instance
is also used for peer-to-peer communication
among the GPUs in multi-node NVLink-connected clusters.
In the multi-node case, communicating GPUs are managed by processes running
within independent operating system instances on each cluster node,
requiring additional abstraction above the level of operating system instances.
Multi-node peer communication is achieved by creating and exchanging
so-called “fabric” handles between multi-node GPU peers, and
by then obtaining process-local device pointers within the participating
processes and operating system instances corresponding
to the multi-node ranks.

See below (single-node CUDA IPC) and ref::*virtual-memory-management* for
the specific APIs used to establish and exchange process-portable and
node and operating system instance-portable handles that are used to obtain
process-local device pointers for GPU communication.

> **Note**
>
> There are individual advantages and limitations associated with the use of the CUDA IPC APIs and Virtual Memory Management (VMM) APIs when used for IPC.
>
> The CUDA IPC API is only currently supported on Linux platforms.
>
> The CUDA Virtual Memory Management APIs permit per-allocation control over
> peer accessibility and sharing at memory allocation time, but require the use of the CUDA Driver API.
