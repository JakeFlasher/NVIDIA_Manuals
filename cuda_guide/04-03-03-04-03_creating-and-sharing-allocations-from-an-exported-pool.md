---
title: "4.3.3.4.3. Creating and Sharing Allocations from an Exported Pool"
section: "4.3.3.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#creating-and-sharing-allocations-from-an-exported-pool"
---

#### [4.3.3.4.3. Creating and Sharing Allocations from an Exported Pool](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-and-sharing-allocations-from-an-exported-pool)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-and-sharing-allocations-from-an-exported-pool "Permalink to this headline")

Once the pool has been shared, allocations made with `cudaMallocAsync()`
from the pool in the exporting process can be shared with processes that
have imported the pool. Since the pool’s security policy is established and
verified at the pool level, the OS does not need extra bookkeeping to provide
security for specific pool allocations. In other words, the opaque
`cudaMemPoolPtrExportData` required to import a pool allocation may be sent
to the importing process using any mechanism.

While allocations may be exported and imported without synchronizing with
the allocating stream in any way, the importing process must follow the same
rules as the exporting process when accessing the allocation. Specifically, access
to the allocation must happen after the allocation
operation in the allocating stream executes. The two following code snippets show
`cudaMemPoolExportPointer()` and `cudaMemPoolImportPointer()` sharing the
allocation with an IPC event used to guarantee that the allocation isn’t
accessed in the importing process before the allocation is ready.

```c++
// preparing an allocation in the exporting process
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// ipc event for coordinating between processes
// cudaEventInterprocess flag makes the event an ipc event
// cudaEventDisableTiming  is set for performance reasons

cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// allocate from the exporting mem pool
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// event for sharing when the allocation is ready.
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share IPC event and pointer export data with the importing process using
//  any mechanism. Here we copy the data into shared memory
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// signal consumers data is ready
```

```c++
// Importing an allocation
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// Need to retrieve the ipc event handle and the export data from the
// exporting process using any mechanism.  Here we are using shmem and just
// need synchronization to make sure the shared memory is filled in.

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// import the allocation. The operation does not block on the allocation being ready.
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// Wait for the prior stream operations in the allocating stream to complete before
// using the allocation in the importing process.
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

When freeing the allocation, the allocation must be freed in the importing
process before it is freed in the exporting process. The following code
snippet demonstrates the use of CUDA IPC events to provide the required
synchronization between the `cudaFreeAsync` operations in both processes.
Access to the allocation from the importing process is obviously restricted by
the free operation in the importing process side. It is worth noting that
`cudaFree` can be used to free the allocation in both processes and that
other stream synchronization APIs may be used instead of CUDA IPC events.

```c++
// The free must happen in importing process before the exporting process
kernel<<<..., stream>>>(ptr, ...);

// Last access in importing process
cudaFreeAsync(ptr, stream);

// Access not allowed in the importing process after the free
cudaIpcEventRecord(finishedIpcEvent, stream);
```

```c++
// Exporting process
// The exporting process needs to coordinate its free with the stream order
// of the importing process’s free.
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// The free in the importing process doesn’t stop the exporting process
// from using the allocation.
cudFreeAsync(ptrInExportingProcess,stream);
```
