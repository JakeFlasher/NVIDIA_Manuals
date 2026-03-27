---
title: "3.4. Programming Systems with Multiple GPUs"
section: "3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#programming-systems-with-multiple-gpus"
---

# [3.4. Programming Systems with Multiple GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#programming-systems-with-multiple-gpus)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#programming-systems-with-multiple-gpus "Permalink to this headline")

Multi-GPU programming allows an application to address problem sizes
and achieve performance levels beyond what is possible with a single GPU
by exploiting the larger aggregate arithmetic performance,
memory capacity, and memory bandwidth provided by multi-GPU systems.

CUDA enables multi-GPU programming through host APIs,
driver infrastructure, and supporting GPU hardware technologies:

- Host thread CUDA context management
- Unified memory addressing for all processors in the system
- Peer-to-peer bulk memory transfers between GPUs
- Fine-grained peer-to-peer GPU load/store memory access
- Higher level abstractions and supporting system software such as CUDA interprocess communication, parallel reductions using [NCCL](https://developer.nvidia.com/nccl), and communication using NVLink and/or GPU-Direct RDMA with APIs such as [NVSHMEM](https://developer.nvidia.com/nvshmem) and MPI

At the most basic level, multi-GPU programming requires the application
to manage multiple active CUDA contexts concurrently,
distribute data to the GPUs,
launch kernels on the GPUs to complete their work,
and to communicate or collect the results so that they can be acted upon
by the application.
The details of how this is done differ depending on the most effective
mapping of an application’s algorithms, available parallelism,
and existing code structure to a suitable multi-GPU programming approach.
Some of the most common multi-GPU programming approaches include:

- A single host thread driving multiple GPUs
- Multiple host threads, each driving their own GPU
- Multiple single-threaded host processes, each driving their own GPU
- Multiple host processes containing multiple threads, each driving their own GPU
- Multi-node NVLink-connected clusters, with GPUs driven by threads and processes running within multiple operating system instances across the cluster nodes

GPUs can communicate with each other
through memory transfers and peer accesses between device memories,
covering each of the multi-device work distribution approaches listed above.
High performance, low-latency GPU communications are supported
by querying for and enabling the use of peer-to-peer GPU memory access,
and leveraging NVLink to achieve high bandwidth transfers and
finer-grained load/store operations between devices.

CUDA unified virtual addressing permits communication between multiple GPUs
within the same host process with minimal additional steps to query and enable
the use of high performance peer-to-peer memory access and transfers,
e.g., via NVLink.

Communication between multiple GPUs managed by different host processes
is supported through the use of interprocess communication (IPC) and
Virtual memory Management (VMM) APIs.
An introduction to high level IPC concepts and intra-node CUDA IPC APIs
are discussed in the [Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication) section.
AdvancedVirtual Memory Management (VMM) APIs support both intra-node and
multi-node IPC, are usable on both Linux and Windows
operating systems, and allow per-allocation granularity control
over IPC sharing of memory buffers as described in
[Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management).

CUDA itself provides the APIs needed to implement collective operations
within a group of GPUs, potentially including the host, but it does not
provide high level multi-GPU collective APIs itself.
Multi-GPU collectives are provided by higher abstraction CUDA communication
libraries such as
[NCCL](https://developer.nvidia.com/nccl)
and
[NVSHMEM](https://developer.nvidia.com/nvshmem).
