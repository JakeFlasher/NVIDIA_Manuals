---
title: "5.4.1.2. Memory Space Specifiers"
section: "5.4.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-space-specifiers"
---

### [5.4.1.2. Memory Space Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#memory-space-specifiers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#memory-space-specifiers "Permalink to this headline")

The memory space specifiers `__device__`, `__managed__`, `__constant__`, and `__shared__` indicate the storage location of a variable on the device.

The following table summarizes the memory space properties:

| Memory Space Specifier | Location | Accessible by | Lifetime | Unique instance |
| --- | --- | --- | --- | --- |
| `__device__` | Device global memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) | Per device |
| `__constant__` | Device constant memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) | Per device |
| `__managed__` | Host and Device (automatic) | Host/Device Threads | Program | Per program |
| `__shared__` | Device (streaming multiprocessor) | Block Threads | Block | Block |
| no specifier | Device (registers) | Single Thread | Single Thread | Single Thread |

---

- Both `__device__` and `__constant__` variables can be accessed from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) functions `cudaGetSymbolAddress()`, `cudaGetSymbolSize()`, `cudaMemcpyToSymbol()`, and `cudaMemcpyFromSymbol()`.
- `__constant__` variables are read-only in device code and can only be modified from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

The following example illustrates how to use these APIs:

```cuda
__device__   float device_var       = 4.0f; // Variable in device memory
__constant__ float constant_mem_var = 4.0f; // Variable in constant memory
                                            // For readability, the following example focuses on a device variable.
int main() {
    float* device_ptr;
    cudaGetSymbolAddress((void**) &device_ptr, device_var);        // Gets address of device_var

    size_t symbol_size;
    cudaGetSymbolSize(&symbol_size, device_var);                   // Retrieves the size of the symbol (4 bytes).

    float host_var;
    cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // Copies from device to host.

    host_var = 3.0f;
    cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // Copies from host to device.
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/vYjP8GGv3).
