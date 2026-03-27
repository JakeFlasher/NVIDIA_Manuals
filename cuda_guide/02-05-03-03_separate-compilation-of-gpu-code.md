---
title: "2.5.3.3. Separate Compilation of GPU Code"
section: "2.5.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#separate-compilation-of-gpu-code"
---

### [2.5.3.3. Separate Compilation of GPU Code](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#separate-compilation-of-gpu-code)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#separate-compilation-of-gpu-code "Permalink to this headline")

`nvcc` defaults to _whole-program compilation_, which expects all GPU code and symbols to be present in the compilation unit that uses them. CUDA device functions may call device functions or access device variables defined in other compilation units, but either the `-rdc=true` or its alias the `-dc` flag must be specified on the `nvcc` command line to enable linking of device code from different compilation units. The ability to link device code and symbols from different compilation units is called _separate compilation_.

Separate compilation allows more flexible code organization, can improve compile time, and can lead to smaller binaries. Separate compilation may involve some build-time complexity compared to whole-program compilation. Performance can be affected by the use of device code linking, which is why it is not used by default.  [Link-Time Optimization (LTO)](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#nvcc-link-time-optimization) can help reduce the performance overhead of separate compilation.

Separate compilation requires the following conditions:

- Non-`const` device variables defined in one compilation unit must be referred to with the `extern` keyword in other compilation units.
- All `const` device variables must be defined and referred to with the `extern` keyword.
- All CUDA source files `.cu` must be compiled with the `-dc` or `-rdc=true` flags.

Host and device functions have external linkage by default and do not require the `extern` keyword. Note that [starting from CUDA 13](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/), `__global__` functions and `__managed__`/`__device__`/`__constant__` variables have internal linkage by default.

In the following example, `definition.cu` defines a variable and a function, while `example.cu` refers to them. Both files are compiled separately and linked into the final binary.

```cuda
// ----- definition.cu -----
extern __device__ int device_variable = 5;
__device__        int device_function() { return 10; }
```

```cuda
// ----- example.cu -----
extern __device__ int  device_variable;
__device__        int device_function();

__global__ void kernel(int* ptr) {
    device_variable = 0;
    *ptr            = device_function();
}
```

```bash
nvcc -dc definition.cu -o definition.o
nvcc -dc example.cu    -o example.o
nvcc definition.o example.o -o program
```
