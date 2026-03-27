---
title: "2.2.3.2.2. Dynamic Allocation of Shared Memory"
section: "2.2.3.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#dynamic-allocation-of-shared-memory"
---

#### [2.2.3.2.2. Dynamic Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#dynamic-allocation-of-shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#dynamic-allocation-of-shared-memory "Permalink to this headline")

To allocate shared memory dynamically, the programmer can specify the desired amount of shared memory per thread block in bytes as the third (and optional) argument to the kernel launch in the triple chevron notation like this `functionName<<<grid, block, sharedMemoryBytes>>>()`.

Then, inside the kernel, the programmer can use the `extern __shared__` specifier to declare a variable that will be allocated dynamically at kernel launch.

```c++
extern __shared__ float sharedArray[];
```

One caveat is that if one wants multiple dynamically allocated shared memory arrays, the single `extern __shared__` must be partitioned manually using pointer arithmetic.  For example, if one wants the equivalent of the following,

```c++
short array0[128];
float array1[64];
int   array2[256];
```

in dynamically allocated shared memory, one could declare and initialize the arrays in the following way:

```c++
extern __shared__ float array[];

short* array0 = (short*)array;
float* array1 = (float*)&array0[128];
int*   array2 =   (int*)&array1[64];
```

Note that pointers need to be aligned to the type they point to, so the following code, for example, does not work since `array1` is not aligned to 4 bytes.

```c++
extern __shared__ float array[];
short* array0 = (short*)array;
float* array1 = (float*)&array0[127];
```
