---
title: "2.1.6. Runtime Initialization"
section: "2.1.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#runtime-initialization"
---

## [2.1.6. Runtime Initialization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#runtime-initialization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#runtime-initialization "Permalink to this headline")

The CUDA runtime creates a [CUDA context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) for each device in the system. This context is the primary context for this device and is initialized at the first runtime function which requires an active context on this device. The context is shared among all the host threads of the application. As part of context creation, the device code is [just-in-time compiled](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) if necessary and loaded into device memory. This all happens transparently. The primary context created by the CUDA runtime can be accessed from the driver API for interoperability as described in [Interoperability between Runtime and Driver APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-interop-with-runtime).

As of CUDA 12.0, the `cudaInitDevice` and `cudaSetDevice` calls initialize the runtime and the primary [context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) associated with the specified device. The runtime will implicitly use device 0 and self-initialize as needed to process runtime API requests if they occur before these calls. This is important when timing runtime function calls and when interpreting the error code from the first call into the runtime. Prior to CUDA 12.0, `cudaSetDevice` would not initialize the runtime.

`cudaDeviceReset` destroys the primary context of the current device. If CUDA runtime APIs are called after the primary context has been destroyed, a new primary context for that device will be created.

> **Note**
>
> The CUDA interfaces use global state that is initialized during host program initiation and destroyed during host program termination. Using any of these interfaces (implicitly or explicitly) during program initiation or termination after main will result in undefined behavior.
>
> As of CUDA 12.0, `cudaSetDevice` explicitly initializes the runtime, if it has not already been initialized, after changing the current device for the host thread. Previous versions of CUDA delayed runtime initialization on the new device until the first runtime call was made after `cudaSetDevice`. Because of this, it is very important to check the return value of `cudaSetDevice` for initialization errors.
>
> The runtime functions from the error handling and version management sections of the reference manual do not initialize the runtime.
