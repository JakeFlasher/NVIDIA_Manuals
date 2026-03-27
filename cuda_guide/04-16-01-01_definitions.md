---
title: "4.16.1.1. Definitions"
section: "4.16.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#definitions"
---

### [4.16.1.1. Definitions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#definitions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#definitions "Permalink to this headline")

**Fabric Memory:**
Fabric memory refers to memory that is accessible over a high-speed
interconnect fabric such as NVIDIA’s NVLink and NVSwitch. This fabric provides
a memory coherence and high-bandwidth communication layer between multiple
GPUs or nodes, enabling them to share memory efficiently as if the memory is
attached to a unified fabric rather than isolated on individual devices.

CUDA 12.4 and later have a VMM allocation handle type
`CU_MEM_HANDLE_TYPE_FABRIC`. On supported platforms and provided the NVIDIA
IMEX daemon is running, this allocation handle type enables sharing allocations
not only intra-node with any communication mechanism, e.g. MPI, but also inter-node.
This allows GPUs in a multi-node NVLink system to map the memory of all
other GPUs part of the same NVLink fabric even if they are in different nodes.

**Memory Handles:**
In VMM, handles are opaque identifiers that represent physical memory
allocations. These handles are central to managing memory in the low-level
CUDA VMM API. They enable flexible control over physical memory objects that
can be mapped into virtual address spaces.
A handle uniquely identifies a physical memory allocation.
Handles serve as an abstract reference to memory resources without exposing
direct pointers. Handles allow operations like exporting and importing memory
across processes or devices, facilitating memory sharing and virtualization.

**IMEX Channels:**
The name IMEX stands for _internode memory exchange_ and is part of NVIDIA’s
solution for GPU-to-GPU communication across different nodes.
IMEX channels are a GPU driver feature that provides user-based memory
isolation in multi-user or multi-node environments within an IMEX domain.
IMEX channels serve as a security and isolation mechanism.

IMEX channels are directly related to the fabric handle and has to be enabled
in multi-node GPU communication. When a GPU allocates
memory and wants to make it accessible to a GPU on a different node, it first
needs to export a handle to that memory. The IMEX channel is used during
this export process to generate a secure fabric handle that can only be
imported by a remote process with the correct channel access.

**Unicast Memory Access:**
Unicast memory access in the context of VMM API refers to the
controlled, direct mapping and access of physical memory to a unique virtual
address range by a specific device or process. Instead of broadcasting access
to multiple devices, unicast memory access means that a particular GPU device
is granted explicit read/write permissions to a reserved virtual address range
that maps to a physical memory allocation.

**Multicast Memory Access:**
Multicast memory access in the context of the VMM API refers to the capability
for a single physical memory allocation or region to be mapped
simultaneously to multiple devices’ virtual address spaces using a multicast
mechanism. This allows data to be efficiently shared in a one-to-many fashion
across multiple GPUs, reducing redundant data transfers and
improving communication efficiency.
NVIDIA’s CUDA VMM API supports creating a multicast object that binds together
physical memory allocations from multiple devices.
