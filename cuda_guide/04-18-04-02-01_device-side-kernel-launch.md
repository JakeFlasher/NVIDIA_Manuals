---
title: "4.18.4.2.1. Device-Side Kernel Launch"
section: "4.18.4.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#device-side-kernel-launch"
---

#### [4.18.4.2.1. Device-Side Kernel Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-side-kernel-launch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-side-kernel-launch "Permalink to this headline")

Kernels may be launched from the device using the standard CUDA <<< >>> syntax:

```c++
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

- `Dg` is of type `dim3` and specifies the dimensions and size of the grid
- `Db` is of type `dim3` and specifies the dimensions and size of each thread block
- `Ns` is of type `size_t` and specifies the number of bytes of shared memory that is dynamically allocated per thread block for this call in addition to statically allocated memory. `Ns` is an optional argument that defaults to 0.
- `S` is of type `cudaStream_t` and specifies the stream associated with this call. The stream must have been allocated in the same grid where the call is being made. `S` is an optional argument that defaults to the NULL stream.
