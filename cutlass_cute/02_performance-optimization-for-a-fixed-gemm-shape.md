---
title: "2. Performance Optimization for a Fixed GEMM Shape"
section: "2"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#performance-optimization-for-a-fixed-gemm-shape"
---

#### [2. Performance Optimization for a Fixed GEMM Shape](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#performance-optimization-for-a-fixed-gemm-shape)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#performance-optimization-for-a-fixed-gemm-shape "Permalink to this headline")

To optimize kernel performance for a specific GEMM problem size:

```bash
cutlass_profiler --kernels=*gemm* --enable-best-kernel-for-fixed-shape --m=6144 --n=6144 --k=6144 --sort-results-flops-per-sec
```

To search optimized kernel performance for a series of GEMM shapes (m, n, k = 1024, 2048):

```bash
cutlass_profiler --kernels=*gemm* --enable-best-kernel-for-fixed-shape --m=1024,2048 --n=1024,2048 --k=1024,2048 --sort-results-flops-per-sec
```

It is worth noting that by enabling exhaustive performance search via `--enable-kernel-performance-search`, a user is still able and responsible to decide parameters like data distribution in argument list, for which a user can choose `--dist=uniform,min:-1,max:1,scale:-1` to initialize a tensor with floating point numbers in uniform distribution. Otherwise, those parameters will be initialized to their default values.

For examples above, one can change the kernel filtering regex according to their own use cases.
