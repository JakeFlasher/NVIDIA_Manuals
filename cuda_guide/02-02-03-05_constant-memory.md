---
title: "2.2.3.5. Constant Memory"
section: "2.2.3.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#constant-memory"
---

### [2.2.3.5. Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#constant-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#constant-memory "Permalink to this headline")

Constant memory has a grid scope and is accessible for the lifetime of the application.  The constant memory resides on the device and is read-only to the kernel.  As such, it must be declared and initialized on the host with the `__constant__` specifier, outside any function.

The `__constant__` memory space specifier declares a variable that:

- Resides in constant memory space,
- Has the lifetime of the CUDA context in which it is created,
- Has a distinct object per device,
- Is accessible from all the threads within the grid and from the host through the runtime library (`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).

The total amount of constant memory can be queried with the `totalConstMem` device property element.

Constant memory is useful for small amounts of data that each thread will use in a read-only fashion.  Constant memory is small relative to other memories, typically 64KB per device.

An example snippet of declaring and using constant memory follows.

```c++
// In your .cu file
__constant__ float coeffs[4];

__global__ void compute(float *out) {
    int idx = threadIdx.x;
    out[idx] = coeffs[0] * idx + coeffs[1];
}

// In your host code
float h_coeffs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
compute<<<1, 10>>>(device_out);
```
