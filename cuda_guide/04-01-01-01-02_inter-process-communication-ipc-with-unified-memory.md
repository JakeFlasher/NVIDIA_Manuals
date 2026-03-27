---
title: "4.1.1.1.2. Inter-Process Communication (IPC) with Unified Memory"
section: "4.1.1.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#inter-process-communication-ipc-with-unified-memory"
---

#### [4.1.1.1.2. Inter-Process Communication (IPC) with Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#inter-process-communication-ipc-with-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#inter-process-communication-ipc-with-unified-memory "Permalink to this headline")

> **Note**
>
> As of now, using IPC with unified memory can have significant performance implications.

Many applications prefer to manage one GPU per process, but still need to use unified memory,
for example for over-subscription, and access it from multiple GPUs.

CUDA IPC ( see [Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication) ) does not support managed memory: handles to this type of memory may not be shared through
any of the mechanisms discussed in this section.
On systems with full CUDA unified memory support, system-allocated memory is IPC capable.
Once access to system-allocated memory has been shared with other processes,
the same programming model applies, similar to [File-backed Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-sam-file-backed).

See the following references for more information on various ways of creating
IPC-capable system-allocated memory under Linux:

- [mmap with MAP_SHARED](https://man7.org/linux/man-pages/man2/mmap.2.html)
- [POSIX IPC APIs](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
- [Linux memfd_create](https://man7.org/linux/man-pages/man2/memfd_create.2.html) .

Note that it is not possible to share memory between different hosts and their devices using this technique.
