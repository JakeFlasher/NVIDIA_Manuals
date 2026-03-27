---
title: "5.4.1.2.1. __shared__ Memory"
section: "5.4.1.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#cpp-language-extensions--shared-memory"
---

#### [5.4.1.2.1. __shared__ Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#shared-memory "Permalink to this headline")

`__shared__` memory variables can have a static size, which is determined at compile time, or a dynamic size, which is determined at kernel launch time. See the [Kernel Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#execution-configuration) section for details on specifying the shared memory size at run time.

Shared memory constraints:

- Variables with a dynamic size must be declared as an external array or as a pointer.
- Variables with a static size cannot be initialized in their declaration.

The following example illustrates how to declare and size `__shared__` variables:

```cuda
extern __shared__ char dynamic_smem_pointer[];
// extern __shared__ char* dynamic_smem_pointer; alternative syntax

__global__ void kernel() { // or a __device__ function
    __shared__ int smem_var1[4];                  // static size
    auto smem_var2 = (int*) dynamic_smem_pointer; // dynamic size
}

int main() {
    size_t shared_memory_size = 16;
    kernel<<<1, 1, shared_memory_size>>>();
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/nPjvd1frb).
