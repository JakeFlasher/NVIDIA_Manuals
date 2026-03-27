---
title: "4.16.4.2. Add Devices to Multicast Objects"
section: "4.16.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#add-devices-to-multicast-objects"
---

### [4.16.4.2. Add Devices to Multicast Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#add-devices-to-multicast-objects)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#add-devices-to-multicast-objects "Permalink to this headline")

Devices can be added to a multicast team with `cuMulticastAddDevice`:

```c++
cuMulticastAddDevice(&mcHandle, device);
```

This step needs to be completed on all processes controlling devices that participate in a multicast team before memory on any device is
bound to the multicast object.
