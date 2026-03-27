---
title: "2.2.4.2. Shared Memory Access Patterns"
section: "2.2.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns"
---

### [2.2.4.2. Shared Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#shared-memory-access-patterns)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#shared-memory-access-patterns "Permalink to this headline")

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle.

When multiple threads in the same warp attempt to access different elements in the same bank, a bank conflict occurs.  In this case, the access to the data in that bank will be serialized until the data in that bank has been obtained by all the threads that have requested it.  This serialization of access results in a performance penalty.

The two exceptions to this scenario happen when multiple threads in the same warp are accessing (either reading or writing) the same shared memory location.  For read accesses, the word is broadcast to the requesting threads.  For write accesses, each shared memory address is written by only one of the threads (which thread performs the write is undefined).

[Figure 13](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory-5-x-examples-of-strided-shared-memory-accesses) shows some examples of strided access.  The red box inside the bank indicates a unique location in shared memory.

![Strided Shared Memory Accesses in 32 bit bank size mode.](images/______-______-______-_________1.png)

Figure 13 Strided Shared Memory Accesses in 32 bit bank size mode.[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory-5-x-examples-of-strided-shared-memory-accesses "Link to this image")

**Left**

  Linear addressing with a stride of one 32-bit word (no bank conflict).

**Middle**

  Linear addressing with a stride of two 32-bit words (two-way bank conflict).

**Right**

  Linear addressing with a stride of three 32-bit words (no bank conflict).

[Figure 14](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory-5-x-examples-of-irregular-shared-memory-accesses) shows some examples of memory read accesses that involve the broadcast mechanism.  The red box inside the bank indicates a unique location in shared memory.  If multiple arrows point to the same location, the data is broadcast to all threads that requested it.

![Irregular Shared Memory Accesses.](images/______-______-______-_________2.png)

Figure 14 Irregular Shared Memory Accesses.[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-shared-memory-5-x-examples-of-irregular-shared-memory-accesses "Link to this image")

**Left**

  Conflict-free access via random permutation.

**Middle**

  Conflict-free access since threads 3, 4, 6, 7, and 9 access the same word within bank 5.

**Right**

  Conflict-free broadcast access (threads access the same word within a bank).

> **Note**
>
> Avoiding bank conflicts is an important performance consideration for writing performant CUDA kernels that use shared memory.
