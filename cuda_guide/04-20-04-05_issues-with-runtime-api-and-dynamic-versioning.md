---
title: "4.20.4.5. Issues with Runtime API and Dynamic Versioning"
section: "4.20.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#issues-with-runtime-api-and-dynamic-versioning"
---

### [4.20.4.5. Issues with Runtime API and Dynamic Versioning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#issues-with-runtime-api-and-dynamic-versioning)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#issues-with-runtime-api-and-dynamic-versioning "Permalink to this headline")

More complications arise when we consider different combinations of the CUDA version with which an application is compiled, CUDA runtime version, and CUDA driver version that an application dynamically links against.

```c++
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;
enum cudaDriverEntryPointQueryResult runtimeStatus;

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriver;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriver, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriver) {
    // pfn_cuDeviceGetUuidDriver points to ???
}

// Ask the runtime for the function
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &runtimeStatus);
if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuidRuntime points to ???
}

// Ask the driver for the function based on the driver version (obtained via runtime)
error = cudaDriverGetVersion(&driverVersion);
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriverDriverVer;
status = cuGetProcAddress ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriverDriverVer, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriverDriverVer) {
    // pfn_cuDeviceGetUuidDriverDriverVer points to ???
}
```

The following matrix of function pointers is expected:

| **Function Pointer** | **Application Compiled/Runtime Dynamic Linked Version/Driver Version** |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **(3 => CUDA 11.3 and 4 => CUDA 11.4)** |  |  |  |  |  |  |  |  |
| **3/3/3** | **3/3/4** | **3/4/3** | **3/4/4** | **4/3/3** | **4/3/4** | **4/4/3** | **4/4/4** |  |
| `pfn_cuDeviceGetUuidDriver` | t1/v1 | t1/v1 | t1/v1 | t1/v1 | N/A | N/A | **t2/v1** | t2/v2 |
| `pfn_cuDeviceGetUuidRuntime` | t1/v1 | t1/v1 | t1/v1 | **t1/v2** | N/A | N/A | **t2/v1** | t2/v2 |
| `pfn_cuDeviceGetUuidDriverDriverVer` | t1/v1 | **t1/v2** | t1/v1 | **t1/v2** | N/A | N/A | **t2/v1** | t2/v2 |

```text
tX -> Typedef version used at compile time
vX -> Version returned/used at runtime
```

If the application is compiled against CUDA Version 11.3, it would have the typedef for the original function, but if compiled against CUDA Version 11.4, it would have the typedef for the _v2 function. Because of that, notice the number of cases where the typedef does not match the actual version returned/used.
