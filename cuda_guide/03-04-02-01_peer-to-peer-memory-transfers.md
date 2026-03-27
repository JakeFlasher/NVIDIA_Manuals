---
title: "3.4.2.1. Peer-to-Peer Memory Transfers"
section: "3.4.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-transfers"
---

### [3.4.2.1. Peer-to-Peer Memory Transfers](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#peer-to-peer-memory-transfers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#peer-to-peer-memory-transfers "Permalink to this headline")

CUDA can perform memory transfers between devices and will take advantage
of dedicated copy engines and NVLink hardware to maximize performance
when peer-to-peer memory access is possible.

`cudaMemcpy` can be used with the copy type `cudaMemcpyDeviceToDevice` or `cudaMemcpyDefault`.

Otherwise, copies must be performed using `cudaMemcpyPeer()`, `cudaMemcpyPeerAsync()`, `cudaMemcpy3DPeer()`, or `cudaMemcpy3DPeerAsync()` as illustrated in the following code sample.

```c++
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0

cudaSetDevice(1);                   // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);              // Allocate memory on device 1

cudaSetDevice(0);                   // Set device 0 as current
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

cudaSetDevice(1);                   // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
```

A copy (in the implicit _NULL_ stream) between the memories of two different devices:

- does not start until all commands previously issued to either device have completed and
- runs to completion before any commands (see [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution)) issued after the copy to either device can start.

Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream.

If peer-to-peer access is enabled between two devices, e.g., as described in [Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#multi-gpu-peer-to-peer-memory-access), peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster.
