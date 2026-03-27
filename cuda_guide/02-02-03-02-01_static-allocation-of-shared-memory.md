---
title: "2.2.3.2.1. Static Allocation of Shared Memory"
section: "2.2.3.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#static-allocation-of-shared-memory"
---

#### [2.2.3.2.1. Static Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#static-allocation-of-shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#static-allocation-of-shared-memory "Permalink to this headline")

To allocate shared memory statically, the programmer must declare a variable inside the kernel using the `__shared__` specifier.  The variable will be allocated in shared memory and will persist for the duration of the kernel execution.  The size of the shared memory declared in this way must be specified at compile time.  For example, the following code snippet, located in the body of the kernel, declares a shared memory array of type `float` with 1024 elements.

```c++
__shared__ float sharedArray[1024];
```

After this declaration, all the threads in the thread block will have access to this shared memory array.  Care must be taken to avoid data races between threads in the same thread block, typically with the use of `__syncthreads()`.
