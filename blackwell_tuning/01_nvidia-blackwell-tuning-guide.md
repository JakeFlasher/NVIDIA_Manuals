---
title: "1. NVIDIA Blackwell Tuning Guide"
section: "1"
source: "https://docs.nvidia.com/cuda/blackwell-tuning-guide/#nvidia-blackwell-tuning-guide"
---

# [1. NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide#nvidia-blackwell-tuning-guide)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#nvidia-blackwell-tuning-guide "Permalink to this headline")

## [1.1. NVIDIA Blackwell GPU Architecture](https://docs.nvidia.com/cuda/blackwell-tuning-guide#nvidia-blackwell-gpu-architecture)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#nvidia-blackwell-gpu-architecture "Permalink to this headline")

The NVIDIA® Blackwell GPU architecture is NVIDIA’s latest architecture for CUDA® compute applications. The NVIDIA Blackwell GPU architecture retains and extends the same CUDA programming model provided by previous NVIDIA GPU architectures such as NVIDIA Ampere GPU architecture and NVIDIA Hopper. Applications that follow the best practices for those architectures should typically see speedups on the Blackwell GPUs without any code changes. This guide summarizes the ways that an application can be fine-tuned to gain additional speedups by leveraging the NVIDIA Blackwell GPU architecture’s features.[^1]

For further details on the programming features discussed in this guide, refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

## [1.2. CUDA Best Practices](https://docs.nvidia.com/cuda/blackwell-tuning-guide#cuda-best-practices)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#cuda-best-practices "Permalink to this headline")

The performance guidelines and best practices described in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) apply to all CUDA-capable GPU architectures. Programmers must primarily focus on following those recommendations to achieve the best performance.

The high-priority recommendations from those guides are as follows:

- Find ways to parallelize sequential code.
- Minimize data transfers between the host and the device.
- Adjust kernel launch configuration to maximize device utilization.
- Ensure that global memory accesses are coalesced.
- Minimize redundant accesses to global memory whenever possible.
- Avoid long sequences of diverged execution by threads within the same warp.

## [1.3. Application Compatibility](https://docs.nvidia.com/cuda/blackwell-tuning-guide#application-compatibility)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#application-compatibility "Permalink to this headline")

Before addressing specific performance tuning issues covered in this guide, refer to the [Blackwell Compatibility Guide for CUDA Applications](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) to ensure that your application is compiled in a way that is compatible with NVIDIA Blackwell.

## [1.4. NVIDIA Blackwell Tuning](https://docs.nvidia.com/cuda/blackwell-tuning-guide#nvidia-blackwell-tuning)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#nvidia-blackwell-tuning "Permalink to this headline")

### [1.4.1. Streaming Multiprocessor](https://docs.nvidia.com/cuda/blackwell-tuning-guide#streaming-multiprocessor)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#streaming-multiprocessor "Permalink to this headline")

The NVIDIA Blackwell Streaming Multiprocessor (SM) provides the following improvements over the NVIDIA Hopper GPU architecture.

#### [1.4.1.1. Occupancy](https://docs.nvidia.com/cuda/blackwell-tuning-guide#occupancy)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#occupancy "Permalink to this headline")

The maximum number of concurrent warps per SM is 64 for compute capability 10.0 and 48 for compute capability 12.0. Other [factors influencing warp occupancy](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator) are:

- The register file size is 64K 32-bit registers per SM.
- The maximum number of registers per thread is 255.
- The maximum number of thread blocks per SM is 32 for devices of compute capability 10.0 and 12.0.
- For devices of compute capability 10.0 shared memory capacity per SM is 228 KB. For devices of compute capability 12.0, shared memory capacity per SM is 128KB.
- For devices of compute capability 10.0 the maximum shared memory per thread block is 227 KB. For devices of compute capability 12.0 the maximum shared memory per thread block is 99 KB.
- For applications using Thread Block Clusters, it is always recommended to compute the occupancy using `cudaOccupancyMaxActiveClusters` and launch cluster-based kernels accordingly.

Overall, developers can expect similar occupancy as on NVIDIA Hopper GPU architecture GPUs without changes to their application.

#### [1.4.1.2. Thread Block Clusters](https://docs.nvidia.com/cuda/blackwell-tuning-guide#thread-block-clusters)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#thread-block-clusters "Permalink to this headline")

NVIDIA Hopper Architecture added a new optional level of hierarchy, Thread Block Clusters, that allows for further possibilities when parallelizing applications. Thread block clusters are supported by Blackwell GPUs as well. A thread block can read from, write to, and perform atomics in shared memory of other thread blocks within its cluster. This is known as Distributed Shared Memory. As demonstrated in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory), there are applications that cannot fit required data within shared memory and must use global memory instead. Distributed shared memory can act as an intermediate step between these two options.

Distributed Shared Memory can be used by an SM simultaneously with L2 cache accesses. This can benefit applications that need to communicate data between SMs by utilizing the combined bandwidth of both distributed shared memory and L2.

In order to achieve best performance for accesses to Distributed Shared Memory,
access patterns to those described in the
[CUDA C++ Best Practices Guide for Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)
should be used.
Specifically, accesses to Distributed Shared Memory should be coalesced
and aligned to 32-byte segments, if possible.
Access patterns with non-unit stride should be avoided if possible,
which can be achieved by using local shared memory,
similar to what is shown in the
[CUDA C++ Best Practices Guide for Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory).

The maximum portable cluster size supported is 8; however, NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16 by opting in. Launching a kernel with a nonportable cluster size requires setting the `cudaFuncAttributeNonPortableClusterSizeAllowed` function attribute. Using larger cluster sizes may reduce the maximum number of active blocks across the GPU (refer to [Occupancy](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#sm-occupancy)).

### [1.4.2. Memory System](https://docs.nvidia.com/cuda/blackwell-tuning-guide#memory-system)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#memory-system "Permalink to this headline")

#### [1.4.2.1. High-Bandwidth Memory HBM3 Subsystem](https://docs.nvidia.com/cuda/blackwell-tuning-guide#high-bandwidth-memory-hbm3-subsystem)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#high-bandwidth-memory-hbm3-subsystem "Permalink to this headline")

The NVIDIA B200 GPU has support for HBM3 and HBM3e memory, with capacity up to 180 GB.

#### [1.4.2.2. Increased L2 Capacity](https://docs.nvidia.com/cuda/blackwell-tuning-guide#increased-l2-capacity)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#increased-l2-capacity "Permalink to this headline")

The NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB.

The NVIDIA Blackwell  architecture allows CUDA users to control the persistence of data in L2 cache similar to the NVIDIA Ampere GPU Architecture. For more information on the persistence of data in L2 cache, refer to the section on managing L2 cache in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#L2_access_intro).

#### [1.4.2.3. Unified Shared Memory/L1/Texture Cache](https://docs.nvidia.com/cuda/blackwell-tuning-guide#unified-shared-memory-l1-texture-cache)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#unified-shared-memory-l1-texture-cache "Permalink to this headline")

The NVIDIA B200 GPU with compute capability 10.0 has the same the maximum capacity of the combined L1 cache, texture cache, and shared memory of 256 KB as the previous NVIDIA Hopper architecture.

In the NVIDIA Blackwell GPU architecture, the portion of the L1 cache dedicated to shared memory (known as the carveout) can be selected at runtime as in previous architectures such as NVIDIA Ampere Architecture and NVIDIA Volta, using `cudaFuncSetAttribute()` with the attribute `cudaFuncAttributePreferredSharedMemoryCarveout`. Both the NVIDIA H100 GPU and the NVIDIA B200 GPU support shared memory capacities of 0, 8, 16, 32, 64, 100, 132, 164, 196 and 228 KB per SM.

CUDA reserves 1 KB of shared memory per thread block. Hence, the B200 GPU enables a single thread block to address up to 227 KB of shared memory. To maintain architectural compatibility, static shared memory allocations remain limited to 48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit. See the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) for details.

Like GPU architectures going back to NVIDIA Ampere Architecture (compute capability 8.x), the NVIDIA Blackwell GPU architecture combines the functionality of the L1 and texture caches into a unified L1/Texture cache which acts as a coalescing buffer for memory accesses, gathering up the data requested by the threads of a warp before delivery of that data to the warp. Another benefit of its union with shared memory, similar to previous architectures, is improvement in terms of both latency and bandwidth.

### [1.4.3. Fifth-Generation NVLink](https://docs.nvidia.com/cuda/blackwell-tuning-guide#fifth-generation-nvlink)[](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#fifth-generation-nvlink "Permalink to this headline")

The fifth generation of NVIDIA’s high-speed NVLink interconnect is implemented in B200 GPUs, which significantly enhances multi-GPU scalability, performance, and reliability with more links per GPU, much faster communication bandwidth, and improved error-detection and recovery features.

NVLink operates transparently within the existing CUDA model. Transfers between NVLink-connected endpoints are automatically routed through NVLink, rather than PCIe. The `cudaDeviceEnablePeerAccess()` API call remains necessary to enable direct transfers (over either PCIe or NVLink) between GPUs. The `cudaDeviceCanAccessPeer()` can be used to determine if peer access is possible between any pair of GPUs.
