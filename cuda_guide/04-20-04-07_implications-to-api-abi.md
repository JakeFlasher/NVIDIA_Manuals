---
title: "4.20.4.7. Implications to API/ABI"
section: "4.20.4.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#implications-to-api-abi"
---

### [4.20.4.7. Implications to API/ABI](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#implications-to-api-abi)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#implications-to-api-abi "Permalink to this headline")

In the above examples using `cuDeviceGetUuid`, the implications of the mismatched API are minimal, and may not be entirely noticeable to many users as the _v2 was added to support Multi-Instance GPU (MIG) mode. So, on a system without MIG, the user might not even realize they are getting a different API.

More problematic is an API which changes its application signature (and hence ABI) such as `cuCtxCreate`. The _v2 version, introduced in CUDA 3.2 is currently used as the default `cuCtxCreate` when using `cuda.h` but now has a newer version introduced in CUDA 11.4 (`cuCtxCreate_v3`). The API signature has been modified as well, and now takes extra arguments. So, in some of the cases above, where the typedef to the function pointer doesn’t match the returned function pointer, there is a chance for non-obvious ABI incompatibility which would lead to undefined behavior.

For example, assume the following code compiled against a CUDA 11.3 toolkit with a CUDA 11.4 driver installed:

```c++
PFN_cuCtxCreate cuUnknown;
CUdriverProcAddressQueryResult driverStatus;

status = cuGetProcAddress("cuCtxCreate", (void**)&cuUnknown, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && cuUnknown) {
    status = cuUnknown(&ctx, 0, dev);
}
```

Running this code where `cudaVersion` is set to anything >=11040 (indicating CUDA 11.4) could have undefined behavior due to not having adequately supplied all the parameters required for the _v3 version of the `cuCtxCreate_v3` API.
