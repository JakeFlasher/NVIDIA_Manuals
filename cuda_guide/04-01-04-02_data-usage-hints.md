---
title: "4.1.4.2. Data Usage Hints"
section: "4.1.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#data-usage-hints"
---

### [4.1.4.2. Data Usage Hints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#data-usage-hints)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#data-usage-hints "Permalink to this headline")

When multiple processors simultaneously access the same data,
`cudaMemAdvise` may be used to hint how the data at
`[devPtr, devPtr + count)` will be accessed:

```c++
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

The example shows how to use `cudaMemAdvise`:

```cuda
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // use data on GPU
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
// test-prefetch-managed-end

static const int maxDevices = 1;
static const int maxOuterLoopIter = 3;
static const int maxInnerLoopIter = 4;

// test-advise-managed-begin
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
```

Where `advice` may take the following values:

- **`cudaMemAdviseSetReadMostly`:**
  This implies that the data is mostly going to be read from and only occasionally written to.
  In general, it allows trading off read bandwidth for write bandwidth on this region.

- **`cudaMemAdviseSetPreferredLocation`:**
  This hint sets the preferred location for the data to be the specified device’s physical memory.
  This hint encourages the system to keep the data at the preferred location, but does not guarantee it.
  Passing in a value of `cudaMemLocationTypeHost` for location.type sets the preferred location as CPU memory.
  Other hints, like `cudaMemPrefetchAsync`, may override this hint and allow the memory to migrate away from its preferred location.

- **`cudaMemAdviseSetAccessedBy`:**
  In some systems, it may be beneficial for performance to establish a
  mapping into memory before accessing the data from a given processor.
  This hint tells the system that the data will be frequently accessed by `location.id`
  when `location.type` is `cudaMemLocationTypeDevice`,
  enabling the system to assume that creating these mappings pays off.
  This hint does not imply where the data should reside,
  but it can be combined with `cudaMemAdviseSetPreferredLocation` to specify that.
  On hardware-coherent systems, this hint switches on access counter migration, see [Access Counter Migration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-access-counters).

Each advice can be also unset by using one of the following values:
`cudaMemAdviseUnsetReadMostly`, `cudaMemAdviseUnsetPreferredLocation` and
`cudaMemAdviseUnsetAccessedBy`.

The example shows how to use `cudaMemAdvise`:

**System Allocator**

```cuda
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // Allocate memory using malloc or cudaMallocManaged
  dataPtr = (char*)malloc(dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  free(dataPtr);
}
```

**Managed**

```cuda
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  cudaFree(dataPtr);
}
```
