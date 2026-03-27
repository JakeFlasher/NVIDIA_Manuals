---
title: "4.20.4.4. Issues with Runtime API Usage"
section: "4.20.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#issues-with-runtime-api-usage"
---

### [4.20.4.4. Issues with Runtime API Usage](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#issues-with-runtime-api-usage)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#issues-with-runtime-api-usage "Permalink to this headline")

The above examples were focused on the issues with the Driver API usage for obtaining the function pointers to driver APIs. Now we will discuss the potential issues with the Runtime API usage for `cudaApiGetDriverEntryPoint`.

We will start by using the Runtime APIs similar to the above.

```c++
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;

// Ask the runtime for the function
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &driverStatus);
if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuid points to ???
}
```

The function pointer in this example is even more complicated than the driver only examples above because there is no control over which version of the function to obtain; it will always get the API for the current CUDA Runtime version. See the following table for more information:

|  | Static Runtime Version Linkage |  |
| --- | --- | --- |
| Driver Version Installed | **V11.3** | **V11.4** |
| **V11.3** | v1 | v1x |
| **V11.4** | v1 | v2 |

```text
V11.3 => 11.3 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
V11.4 => 11.4 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
v1 => cuDeviceGetUuid
v2 => cuDeviceGetUuid_v2

x => Implies the typedef function pointer won't match the returned
     function pointer.  In these cases, the typedef at compile time
     using a CUDA 11.4 runtime, would match the _v2 version, but the
     returned function pointer would be the original (non _v2) function.
```

The problem in the table comes in with a newer CUDA 11.4 Runtime and Toolkit and older driver (CUDA 11.3) combination, labeled as v1x in the above. This combination would have the driver returning the pointer to the older function (non _v2), but the typedef used in the application would be for the new function pointer.
