---
title: "4.19.2.2.6. Signaling/Waiting on Imported Synchronization Objects"
section: "4.19.2.2.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#signaling-waiting-on-imported-synchronization-objects-dir3d-12-int"
---

#### [4.19.2.2.6. Signaling/Waiting on Imported Synchronization Objects](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#signaling-waiting-on-imported-synchronization-objects-dir3d-12-int)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#signaling-waiting-on-imported-synchronization-objects-dir3d-12-int "Permalink to this headline")

Once the semaphores with fences have been imported from Direct3D12 they can be signaled and waited on.

Signaling a fence object sets its value. The corresponding wait that waits on this signal must be issued in Direct3D12.
Note that the wait that waits on this signal must be issued after this signal has been issued.

```cuda
void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

A fence object waits until its value becomes equal or greater than to the specified value. The corresponding signal that it is waiting on must be issued in Direct3D12.
Note that, the signal must be issued before this wait can be issued.

```cuda
void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```
