---
title: "4.16.2. API Overview"
section: "4.16.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#api-overview"
---

## [4.16.2. API Overview](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#api-overview)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#api-overview "Permalink to this headline")

The VMM API provides developers
with granular control over virtual memory management. VMM, being a very low-level API,
requires use of the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api) directly.
This versatile API can be used in both single-node and multi-node
environments.

To use VMM effectively,
developers must have a solid grasp of a few key concepts in memory management:
- Knowledge of the operating system’s virtual memory fundamentals, including how it handles pages and address spaces
- An understanding of memory hierarchy and hardware characteristics is necessary
- Familiarity with inter-process communication (IPC) methods, such as sockets or message passing,
- A basic knowledge of security for memory access rights

![VMM Usage Overview Diagram](images/___-_______w_1.png)

Figure 52 VMM Usage Overview.
This diagram outlines the series of steps required for VMM utilization.
The process begins by evaluating the environmental setup. Based on this
assessment, the user must make a critical initial decision: whether to
utilize fabric memory handles or OS-specific handles.
A distinct series of subsequent steps must be taken based on the initial
handle choice. However, the final memory management operations—specifically
mapping, reserving, and setting access rights of the allocated memory—are
identical to the type of handle that was selected.[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#vmm-overview-diagram "Link to this image")

The VMM API workflow involves a sequence of steps for memory management, with
a key focus on sharing memory between different devices or processes.
Initially, a developer must allocate physical memory on the source device. To
facilitate sharing, the VMM API utilizes handles to convey necessary
information to the target device or process. The user must export a handle for
sharing, which can be either an OS-specific handle or a fabric-specific
handle. OS-specific handles are limited to inter-process communication on a
single node, while fabric-specific handles offer greater versatility and can
be used in both single-node and multi-node environments. It’s important to
note that using fabric-specific handles requires the enablement of IMEX
channels.

Once the handle is exported, it must be shared with the receiving process or
processes using an inter-process communication protocol, with the choice
of method left to the developer. The receiving process then uses the VMM API
to import the handle. After the handle has been successfully exported, shared,
and imported, both the source and target processes must reserve virtual
address space where the allocated physical memory will be mapped. The final
step is to set the memory access rights for each device, ensuring proper
permissions are established. This entire process, including both handle
approaches, is further detailed in the accompanying figure.
