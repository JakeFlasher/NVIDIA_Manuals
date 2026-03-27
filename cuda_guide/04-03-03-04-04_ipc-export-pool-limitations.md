---
title: "4.3.3.4.4. IPC Export Pool Limitations"
section: "4.3.3.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#ipc-export-pool-limitations"
---

#### [4.3.3.4.4. IPC Export Pool Limitations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ipc-export-pool-limitations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ipc-export-pool-limitations "Permalink to this headline")

IPC pools currently do not support releasing physical blocks back to the OS.
As a result the `cudaMemPoolTrimTo` API has no effect and the
`cudaMemPoolAttrReleaseThreshold` is effectively ignored. This behavior is
controlled by the driver, not the runtime and may change in a future driver
update.
