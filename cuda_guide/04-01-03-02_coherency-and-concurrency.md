---
title: "4.1.3.2. Coherency and Concurrency"
section: "4.1.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#coherency-and-concurrency"
---

### [4.1.3.2. Coherency and Concurrency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#coherency-and-concurrency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#coherency-and-concurrency "Permalink to this headline")

To ensure coherency the unified memory programming model puts constraints on data accesses while both the CPU and GPU are executing concurrently. In effect, the GPU has exclusive access to all managed data and the CPU is not permitted to access it,
while any kernel operation is executing, regardless of whether the specific kernel is actively using the data.
Concurrent CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault because the page is considered inaccessible to the CPU.

For example the following code runs successfully on devices of compute capability 6.x due to the GPU page faulting capability which lifts all restrictions on simultaneous access
but fails on on pre-6.x architectures and Windows platforms because the GPU program kernel is still active when the CPU touches `y`:

```c++
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    y = 20;            // Error on GPUs not supporting concurrent access

    cudaDeviceSynchronize();
    return  0;
}
```

The program must explicitly synchronize with the GPU before accessing `y` (regardless of whether the GPU kernel actually touches `y`  (or any managed data at all):

```c++
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    y = 20;            //  Success on GPUs not supporting concurrent access
    return  0;
}
```

Note that any function call that logically guarantees the GPU completes its work is valid to ensure logically that the GPU work is completed, see [Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#advanced-host-explicit-synchronization).

Note that if memory is dynamically allocated with `cudaMallocManaged()` or `cuMemAllocManaged()` while the GPU is active, the behavior of the memory is unspecified until additional work is launched or the GPU is synchronized.
Attempting to access the memory on the CPU during this time may or may not cause a segmentation fault. This does not apply to memory allocated using the flag `cudaMemAttachHost` or `CU_MEM_ATTACH_HOST`.
