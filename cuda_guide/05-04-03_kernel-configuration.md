---
title: "5.4.3. Kernel Configuration"
section: "5.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#kernel-configuration"
---

## [5.4.3. Kernel Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#kernel-configuration)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#kernel-configuration "Permalink to this headline")

Any call to a `__global__` function must specify an _execution configuration_ for that call. This execution configuration defines the dimensions of the grid and blocks that will be used to execute the function on the device, as well as the associated [stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams).

The execution configuration is specified by inserting an expression in the form `<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>` between the function name and the parenthesized argument list, where:

- `grid_dim` is of type [dim3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-variables) and specifies the dimension and size of the grid, such that `grid_dim.x * grid_dim.y * grid_dim.z` equals the number of blocks being launched;
- `block_dim` is of type [dim3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-variables) and specifies the dimension and size of each block, such that `block_dim.x * block_dim.y * block_dim.z` equals the number of threads per block;
- `dynamic_smem_bytes` is an optional `size_t` argument that defaults to zero. It specifies the number of bytes in shared memory that are dynamically allocated per block for this call in addition to the statically allocated memory. This memory is used by `extern __shared__` arrays (see [__shared__ Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#shared-memory-specifier)).
- `stream` is of type `cudaStream_t` (pointer) and specifies the associated stream. `stream` is an optional argument that defaults to `NULL`.

The following example shows a kernel function declaration and call:

```cuda
__global__ void kernel(float* parameter);

kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);
```

The arguments for the execution configuration are evaluated before the arguments for the actual function.

The function call fails if `grid_dim` or `block_dim` exceeds the maximum sizes allowed for the device, as specified in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities), or if `dynamic_smem_bytes` is greater than the available shared memory after accounting for statically allocated memory.
