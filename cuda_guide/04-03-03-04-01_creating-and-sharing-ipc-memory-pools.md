---
title: "4.3.3.4.1. Creating and Sharing IPC Memory Pools"
section: "4.3.3.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#creating-and-sharing-ipc-memory-pools"
---

#### [4.3.3.4.1. Creating and Sharing IPC Memory Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-and-sharing-ipc-memory-pools)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-and-sharing-ipc-memory-pools "Permalink to this headline")

Sharing access to a pool involves retrieving an OS-native handle to the pool
with  `cudaMemPoolExportToShareableHandle()`, transferring the
handle to the importing process using OS-native IPC mechanisms, and then
creating an imported memory pool with the
`cudaMemPoolImportFromShareableHandle()` API. For
`cudaMemPoolExportToShareableHandle` to succeed, the memory pool must have been
created with the requested handle type specified in the pool properties
structure.

Please reference [samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC)
for the appropriate IPC mechanisms to
transfer the OS-native handle between processes. The rest of the procedure can
be found in the following code snippets.

```c++
// in exporting process
// create an exportable IPC capable pool on device 0
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// FD based handles are integer types
int fdHandle = 0;

// Retrieve an OS native handle to the pool.
// Note that a pointer to the handle memory is passed in here.
cudaMemPoolExportToShareableHandle(&fdHandle,
             memPool,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

// The handle must be sent to the importing process with the appropriate
// OS-specific APIs.
```

```c++
// in importing process
 int fdHandle;
// The handle needs to be retrieved from the exporting process with the
// appropriate OS-specific APIs.
// Create an imported pool from the shareable handle.
// Note that the handle is passed by value here.
cudaMemPoolImportFromShareableHandle(&importedMemPool,
          (void*)fdHandle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
          0);
```
