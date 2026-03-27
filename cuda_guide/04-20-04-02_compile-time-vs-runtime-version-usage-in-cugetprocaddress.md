---
title: "4.20.4.2. Compile Time vs Runtime Version Usage in cuGetProcAddress"
section: "4.20.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#compile-time-vs-runtime-version-usage-in-cugetprocaddress"
---

### [4.20.4.2. Compile Time vs Runtime Version Usage in cuGetProcAddress](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#compile-time-vs-runtime-version-usage-in-cugetprocaddress)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#compile-time-vs-runtime-version-usage-in-cugetprocaddress "Permalink to this headline")

Let’s take the same issue and make one small tweak. The last example used the compile time constant of CUDA_VERSION to determine which function pointer to obtain. More complications arise if the user queries the driver version dynamically using `cuDriverGetVersion` or `cudaDriverGetVersion` to pass to `cuGetProcAddress`. Example:

```c++
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // Get device 0
// handle status

status = cuDriverGetVersion(&cudaVersion);
// handle status

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
    // pfn_cuDeviceGetUuid points to ???
}
```

In this example, assume the user is compiling with CUDA 11.3. The user would debug, test, and deploy this application with the known behavior of getting `cuDeviceGetUuid` (not the _v2 version). Since CUDA has guaranteed ABI compatibility between minor versions, this same application is expected to run after the driver is upgraded to CUDA 11.4 (without updating the toolkit and runtime) without requiring recompilation. This will have undefined behavior though, because now the typedef for `PFN_cuDeviceGetUuid` will still be of the signature for the original version, but since `cudaVersion` would now be 11040 (CUDA 11.4), `cuGetProcAddress` would return the function pointer to the _v2 version, meaning calling it might have undefined behavior.

Note in this case the original (not the _v2 version) typedef looks like:

```c++
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
```

But the _v2 version typedef looks like:

```c++
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
```

So in this case, the API/ABI is going to be the same and the runtime API call will likely not cause issues–only the potential for unknown uuid return. In [Implications to API/ABI](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#implications-to-api-abi), we discuss a more problematic case of API/ABI compatibility.
