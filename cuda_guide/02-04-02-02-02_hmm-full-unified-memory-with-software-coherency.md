---
title: "2.4.2.2.2. HMM - Full Unified Memory with Software Coherency"
section: "2.4.2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#hmm-full-unified-memory-with-software-coherency"
---

#### [2.4.2.2.2. HMM - Full Unified Memory with Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#hmm-full-unified-memory-with-software-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#hmm-full-unified-memory-with-software-coherency "Permalink to this headline")

_Heterogeneous Memory Management_ (HMM) is a feature available on Linux operating systems (with appropriate kernel versions) which enables software-coherent [full unified memory support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory-full). Heterogeneous memory management brings some of the capabilities and convenience provided by ATS to PCIe-connected GPUs.

On Linux with at least Linux Kernel 6.1.24, 6.2.11, or 6.3 or later, heterogeneous memory management (HMM) may be available. The following command can be used to find if the addressing mode is `HMM`.

```c++
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```

When HMM is available, [full unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory-full) is supported and all system allocations are implicitly unified memory. If a system also has [ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services), HMM is disabled and ATS is used, since ATS provides all the capabilities of HMM and more.
