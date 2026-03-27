---
title: "2.4.3.1. Mapped Memory"
section: "2.4.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory"
---

### [2.4.3.1. Mapped Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#mapped-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#mapped-memory "Permalink to this headline")

On systems with [HMM](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management) or [ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services), all host memory is directly accessible from the GPU using the host pointers.  When ATS or HMM are not available, host allocations can be made accessible to the GPU by _mapping_ the memory into the GPU’s memory space. Mapped memory is always page-locked.

The code examples which follow will illustrate the following array copy kernel operating directly on mapped host memory.

```cuda
__global__ void copyKernel(float* a, float* b)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        a[idx] = b[idx];
}
```

While mapped memory may be useful in some cases where certain data which is not copied to the GPU needs to be accessed from a kernel, accessing mapped memory in a kernel requires transactions across the CPU-GPU interconnect, PCIe, or NVLink C2C. These operations have higher latency and lower bandwidth compared to accessing device memory. Mapped memory should not be considered a performant alternative to [unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory) or [explicit memory management](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-explicit-memory-management) for the majority of a kernel’s memory needs.
