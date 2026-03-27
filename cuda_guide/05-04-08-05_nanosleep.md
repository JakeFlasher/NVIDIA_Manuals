---
title: "5.4.8.5. __nanosleep()"
section: "5.4.8.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nanosleep"
---

### [5.4.8.5. __nanosleep()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nanosleep)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nanosleep "Permalink to this headline")

```cuda
__device__ void __nanosleep(unsigned nanoseconds);
```

The function `__nanosleep(ns)` suspends the thread for a sleep duration of approximately `ns` nanoseconds. The maximum sleep duration is approximately one millisecond.

Example:

The following code implements a mutex with exponential back-off.

```cuda
__device__ void mutex_lock(unsigned* mutex) {
    unsigned ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_unlock(unsigned *mutex) {
    atomicExch(mutex, 0);
}
```
