---
title: "4.16.4.3. Bind Memory to Multicast Objects"
section: "4.16.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#bind-memory-to-multicast-objects"
---

### [4.16.4.3. Bind Memory to Multicast Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#bind-memory-to-multicast-objects)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#bind-memory-to-multicast-objects "Permalink to this headline")

After a multicast object has been created and all participating devices have been added to the multicast object it needs to be backed with
physical memory allocated with `cuMemCreate` for each device:

```c++
cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
```
