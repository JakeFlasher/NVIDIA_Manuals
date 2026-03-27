---
title: "5.3.9.5.2. External Linkage"
section: "5.3.9.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#external-linkage"
---

#### [5.3.9.5.2. External Linkage](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#external-linkage)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#external-linkage "Permalink to this headline")

Device variables or functions with external linkage require [separate compilation mode](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation) across multiple translation units.

In separate compilation mode, if a `__device__` or `__global__` function definition is required to exist in a particular translation unit, then the parameters and return types of the function must be complete in that translation unit. The concept is also known as One Definition Rule-use, or ODR-use.

Example:

```cuda
//first.cu:
struct S;                   // forward declaration
__device__ void foo(S);     // ERROR, type 'S' is an incomplete type
__device__ auto* ptr = foo; // ODR-use, address taken

int main() {}
```

```cuda
//second.cu:
struct S {};               // struct definition
__device__ void foo(S) {}  // function definition
```

```console
# compiler invocation
$ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                 first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```
