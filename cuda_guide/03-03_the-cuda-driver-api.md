---
title: "3.3. The CUDA Driver API"
section: "3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#the-cuda-driver-api"
---

# [3.3. The CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#the-cuda-driver-api)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#the-cuda-driver-api "Permalink to this headline")

Previous sections of this guide have covered the CUDA runtime. As mentioned in [CUDA Runtime API and CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-driver-and-runtime), the CUDA runtime is written on top of the lower level CUDA driver API. This section covers some of the differences between the CUDA runtime and the driver APIs, as well has how to intermix them. Most applications can operate at full performance without ever needing to interact with the CUDA driver API. However, new interfaces are sometimes available in the driver API earlier than the runtime API, and some advanced interfaces, such as [Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management), are only exposed in the driver API.

The driver API is implemented in the `cuda` dynamic library (`cuda.dll` or `cuda.so`) which is copied on the system during the installation of the device driver. All its entry points are prefixed with cu.

It is a handle-based, imperative API: Most objects are referenced by opaque handles that may be specified to functions to manipulate the objects.

The objects available in the driver API are summarized in [Table 6](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#driver-api-objects-available-in-cuda-driver-api).

| Object | Handle | Description |
| --- | --- | --- |
| Device | CUdevice | CUDA-enabled device |
| Context | CUcontext | Roughly equivalent to a CPU process |
| Module | CUmodule | Roughly equivalent to a dynamic library |
| Function | CUfunction | Kernel |
| Heap memory | CUdeviceptr | Pointer to device memory |
| CUDA array | CUarray | Opaque container for one-dimensional or two-dimensional data on the device, readable via texture or surface references |
| Texture object | CUtexref | Object that describes how to interpret texture memory data |
| Surface reference | CUsurfref | Object that describes how to read or write CUDA arrays |
| Stream | CUstream | Object that describes a CUDA stream |
| Event | CUevent | Object that describes a CUDA event |

The driver API must be initialized with `cuInit()` before any function from the driver API is called. A CUDA context must then be created that is attached to a specific device and made current to the calling host thread as detailed in [Context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#driver-api-context).

Within a CUDA context, kernels are explicitly loaded as PTX or binary objects by the host code as described in [Module](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#driver-api-module). Kernels written in C++ must therefore be compiled separately into _PTX_ or binary objects. Kernels are launched using API entry points as described in [Kernel Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#driver-api-kernel-execution).

Any application that wants to run on future device architectures must load _PTX_, not binary code. This is because binary code is architecture-specific and therefore incompatible with future architectures, whereas _PTX_ code is compiled to binary code at load time by the device driver.

Here is the host code of the sample from [Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#kernels) written using the driver API:

```c++
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Initialize
    cuInit(0);

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit (0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Create module from binary file
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "VecAdd.ptx");

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);
    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);
    CUdeviceptr d_C;
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // Get function handle from module
    CUfunction vecAdd;
    cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(vecAdd,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

    ...
}
```

Full code can be found in the `vectorAddDrv` CUDA sample.
