---
title: "4.1.1.2.3. Host Native Atomics"
section: "4.1.1.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#host-native-atomics"
---

#### [4.1.1.2.3. Host Native Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#host-native-atomics)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#host-native-atomics "Permalink to this headline")

Some devices, including NVLink-connected devices of hardware-coherent systems, support hardware-accelerated atomic accesses to CPU-resident memory. This implies that atomic accesses to host memory do not have to be emulated with a page fault. For these devices, the attribute `cudaDevAttrHostNativeAtomicSupported` is set to 1.
