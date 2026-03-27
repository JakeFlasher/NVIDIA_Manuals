---
title: "4.19.2.2.1. Matching Device LUIDs"
section: "4.19.2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#matching-device-luids"
---

#### [4.19.2.2.1. Matching Device LUIDs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#matching-device-luids)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#matching-device-luids "Permalink to this headline")

When importing memory and synchronization objects exported by Direct3D12,  they must be imported and mapped on the same device as they were created on.
The CUDA device that corresponds to the Direct3D12 device on which the objects were created can be determined by comparing the LUID of a CUDA device with that of the Direct3D12 device,
as shown in the following code sample.
Note that the Direct3D12 device must not be created on a linked node adapter, i.e. the node count as returned by `ID3D12Device::GetNodeCount` must be 1.

```cuda
int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
    LUID d3d12Luid = d3d12Device->GetAdapterLuid();

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        char *cudaLuid = deviceProp.luid;

        if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
            !memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}
```
