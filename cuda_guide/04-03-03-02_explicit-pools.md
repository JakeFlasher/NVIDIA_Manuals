---
title: "4.3.3.2. Explicit Pools"
section: "4.3.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#explicit-pools"
---

### [4.3.3.2. Explicit Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#explicit-pools)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#explicit-pools "Permalink to this headline")

`cudaMemPoolCreate` creates an explicit pool. This allows
applications to request properties for their allocation beyond what is
provided by the default/implicit pools. These include properties such as IPC
capability, maximum pool size, allocations resident on a specific CPU NUMA
node on supported platforms etc.

```c++
// create a pool similar to the implicit pool on device 0
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));
```

The following code snippet illustrates an example of creating an IPC capable
memory pool on a valid CPU NUMA node.

```c++
// create a pool resident on a CPU NUMA node that is capable of IPC sharing (via a file descriptor).
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&ipcMemPool, &poolProps));
```
