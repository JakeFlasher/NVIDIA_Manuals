---
title: "4.1.1.2.2. Direct Unified Memory Access from the Host"
section: "4.1.1.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#direct-unified-memory-access-from-the-host"
---

#### [4.1.1.2.2. Direct Unified Memory Access from the Host](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#direct-unified-memory-access-from-the-host)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#direct-unified-memory-access-from-the-host "Permalink to this headline")

Some devices have hardware support for coherent reads, stores and atomic accesses
from the host on GPU-resident unified memory.
These devices have the attribute `cudaDevAttrDirectManagedMemAccessFromHost` set to 1.
Note that all hardware-coherent systems have this attribute set for NVLink-connected devices.
On these systems, the host has direct access to GPU-resident memory without page faults and
data migration. Note that with CUDA managed memory, the `cudaMemAdviseSetAccessedBy` hint with location type `cudaMemLocationTypeHost` is necessary
to enable this direct access without page faults, see example below.

**System Allocator**

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // for shared page table systems, the following hint is not necesary
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  free(ret);
}
```

**Managed**

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // set direct access hint

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  cudaFree(ret);
```

After `write` kernel is completed, `ret` will be created and initialized in GPU memory.
Next, the CPU will access `ret` followed by `append` kernel using the same `ret` memory again.
This code will show different behavior depending on the system architecture and support of hardware coherency:

- on systems with `directManagedMemAccessFromHost=1`:
CPU accesses to the managed buffer will not trigger any migrations;
the data will remain resident in GPU memory and any subsequent GPU kernels
can continue to access it directly without inflicting faults or migrations
- on systems with `directManagedMemAccessFromHost=0`:
CPU accesses to the managed buffer will page fault and initiate data migration;
any GPU kernel trying to access the same data first time will page fault and
migrate pages back to GPU memory.
