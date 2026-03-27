---
title: "4.16.5.4. OS-Specific Handle Details for IPC"
section: "4.16.5.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#os-specific-handle-details-for-ipc"
---

### [4.16.5.4. OS-Specific Handle Details for IPC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#os-specific-handle-details-for-ipc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#os-specific-handle-details-for-ipc "Permalink to this headline")

With `cuMemCreate`, users have can indicate at
allocation time that they have earmarked a particular allocation for inter-process communication
or graphics interop purposes. Applications can do this
by setting `CUmemAllocationProp::requestedHandleTypes` to a
platform-specific field. On Windows, when
`CUmemAllocationProp::requestedHandleTypes` is set to
`CU_MEM_HANDLE_TYPE_WIN32` applications must also specify an
LPSECURITYATTRIBUTES attribute in
`CUmemAllocationProp::win32HandleMetaData`. This security attribute defines
the scope of which exported allocations may be transferred to other processes.

Users must ensure they query for support of the requested handle type before
attempting to export memory allocated with `cuMemCreate`. The following
code snippet illustrates query for handle type support in a platform-specific
way.

```c++
int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
```

Users should set the `CUmemAllocationProp::requestedHandleTypes` appropriately as shown below:

```c++
#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif
```
