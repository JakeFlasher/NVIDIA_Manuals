---
title: "5.1.3. Features and Technical Specifications"
section: "5.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications"
---

## [5.1.3. Features and Technical Specifications](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#features-and-technical-specifications)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#features-and-technical-specifications "Permalink to this headline")

| **Feature Support** | **Compute Capability** |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| (Unlisted features are supported for all compute capabilities) | 7.x | 8.x | 9.0 | 10.x | 11.0 | 12.x |
| Atomic functions operating on 128-bit integer values in shared and global memory ([Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions)) | No | Yes |  |  |  |  |
| Atomic addition operating on `float2` and `float4` floating point vectors in global memory ([atomicAdd()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicadd)) | No | Yes |  |  |  |  |
| Warp reduce functions ([Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions)) | No | Yes |  |  |  |  |
| Bfloat16-precision floating-point operations | No | Yes |  |  |  |  |
| 128-bit-precision floating-point operations | No | Yes |  |  |  |  |
| Hardware-accelerated `memcpy_async` ([Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines)) | No | Yes |  |  |  |  |
| Hardware-accelerated Split Arrive/Wait Barrier ([Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers)) | No | Yes |  |  |  |  |
| L2 Cache Residency Management ([L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#advanced-kernels-l2-control)) | No | Yes |  |  |  |  |
| DPX Instructions for Accelerated Dynamic Programming ([Dynamic Programming eXtension (DPX) Instructions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#dpx-instructions)) | Multiple Instr. | Native | Multiple Instr. |  |  |  |
| Distributed Shared Memory | No | Yes |  |  |  |  |
| Thread Block Cluster ([Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters)) | No | Yes |  |  |  |  |
| Tensor Memory Accelerator (TMA) unit ([Using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-tma)) | No | Yes |  |  |  |  |

Note that the KB and K units used in the following tables correspond to 1024 bytes (i.e., a KiB) and 1024 respectively.

|  | **Compute Capability** |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 10.3 | 11.0 | 12.x |
| Ratio of FP32 to FP64 Throughput [^[2]] | 32:1 | 2:1 | 64:1 | 2:1 | 64:1 |  |  |  |  |  |
| Maximum number of resident grids per device (Concurrent Kernel Execution) | 128 |  |  |  |  |  |  |  |  |  |
| Maximum dimensionality of a grid | 3 |  |  |  |  |  |  |  |  |  |
| Maximum x-dimension of a grid | 2<sup>31</sup>-1 |  |  |  |  |  |  |  |  |  |
| Maximum y- or z-dimension of a grid | 65535 |  |  |  |  |  |  |  |  |  |
| Maximum dimensionality of a thread block | 3 |  |  |  |  |  |  |  |  |  |
| Maximum x- or y-dimensionality of a thread block | 1024 |  |  |  |  |  |  |  |  |  |
| Maximum z-dimension of a thread block | 64 |  |  |  |  |  |  |  |  |  |
| Maximum number of threads per block | 1024 |  |  |  |  |  |  |  |  |  |
| Warp size | 32 |  |  |  |  |  |  |  |  |  |
| Maximum number of resident blocks per SM | 16 | 32 | 16 | 24 | 32 | 24 |  |  |  |  |
| Maximum number of resident warps per SM | 32 | 64 | 48 | 64 | 48 |  |  |  |  |  |
| Maximum number of resident threads per SM | 1024 | 2048 | 1536 | 2048 | 1536 |  |  |  |  |  |
| Green contexts: minimum SM partition size for useFlags 0 | 2 | 4 | 8 |  |  |  |  |  |  |  |
| Green contexts: SM co-scheduled alignment per partition for useFlags 0 | 2 | 8 |  |  |  |  |  |  |  |  |

[^[2]]: Non-Tensor Core throughputs.  For more information on throughput see the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)

|  | **Compute Capability** |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x | 11.0 | 12.x |
| Number of 32-bit registers per SM | 64 K |  |  |  |  |  |  |  |  |
| Maximum number of 32-bit registers per thread block | 64 K |  |  |  |  |  |  |  |  |
| Maximum number of 32-bit registers per thread | 255 |  |  |  |  |  |  |  |  |
| Maximum amount of shared memory per SM | 64 KB | 164 KB | 100 KB | 164 KB | 100 KB | 228 KB | 100 KB |  |  |
| Maximum amount of shared memory per thread block [^[3]] | 64 KB | 163 KB | 99 KB | 163 KB | 99 KB | 227 KB | 99 KB |  |  |
| Number of shared memory banks | 32 |  |  |  |  |  |  |  |  |
| Maximum amount of local memory per thread | 512 KB |  |  |  |  |  |  |  |  |
| Constant memory size | 64 KB |  |  |  |  |  |  |  |  |
| Cache working set per SM for constant memory | 8 KB |  |  |  |  |  |  |  |  |
| Cache  working set per SM for texture memory | 32 or 64 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 256 KB | 28 KB ~ 128 KB |  |  |

[^[3]]: Kernels relying on shared memory allocations over 48 KB per block must use dynamic shared memory and require an explicit opt-in, see [Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernel-l1-shared-config).

| Compute Capability | Unified Data Cache Size (KB) | SMEM Capacity Sizes (KB) |
| --- | --- | --- |
| 7.5 | 96 | 32, 64 |
| 8.0 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.6 | 128 | 0, 8, 16, 32, 64, 100 |
| 8.7 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.9 | 128 | 0, 8, 16, 32, 64, 100 |
| 9.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 10.x | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 11.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 12.x | 128 | 0, 8, 16, 32, 64, 100 |

[Table 33](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compute-capabilities-table-tensor-core-data-types-per-compute-capability) shows the input data types supported by Tensor Core acceleration.  The Tensor Core feature set is available within the CUDA compilation toolchain through inline PTX.  It is strongly recommended that applications use this feature set through CUDA-X libraries such as cuDNN, cuBLAS, and cuFFT, for example, or through [CUTLASS](https://docs.nvidia.com/cutlass/index.html), a collection of CUDA C++ template abstractions and Python domain-specific languages (DSLs) designed to enable high-performance matrix-matrix multiplication (GEMM) and related computations across all levels within CUDA.

| Compute Capability | Tensor Core Input Data Types |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | FP64 | TF32 | BF16 | FP16 | FP8 | FP6 | FP4 | INT8 | INT4 |
| 7.5 |  | Yes |  | Yes | Yes |  |  |  |  |
| 8.0 | Yes | Yes | Yes | Yes |  | Yes | Yes |  |  |
| 8.6 |  | Yes | Yes | Yes |  | Yes | Yes |  |  |
| 8.7 |  | Yes | Yes | Yes |  | Yes | Yes |  |  |
| 8.9 |  | Yes | Yes | Yes | Yes |  | Yes | Yes |  |
| 9.0 | Yes | Yes | Yes | Yes | Yes |  | Yes |  |  |
| 10.0 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 10.3 |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 11.0 |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
| 12.x |  | Yes | Yes | Yes | Yes | Yes | Yes | Yes |  |
