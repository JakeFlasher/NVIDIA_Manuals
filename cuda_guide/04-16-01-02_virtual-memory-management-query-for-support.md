---
title: "4.16.1.2. Query for Support"
section: "4.16.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management--query-for-support"
---

### [4.16.1.2. Query for Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#query-for-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#query-for-support "Permalink to this headline")

Applications should query for feature support before attempting to use
them, as their availability can vary depending on the GPU architecture, driver
version, and specific software libraries being used. The following sections
detail how to programmatically check for the necessary support.

**VMM Support**
Before attempting to use VMM APIs, applications must
ensure that the devices they want to use support CUDA virtual memory
management. The following code sample shows querying for VMM support:

```c++
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management
}
```

**Fabric Memory Support:**
Before attempting to use fabric memory, applications must ensure that the
devices they want to use support fabric memory. The following code
sample shows querying for fabric memory support:

```c++
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` supports Fabric Memory
}
```

Aside from using `CU_MEM_HANDLE_TYPE_FABRIC` as handle type and not
requiring OS native mechanisms for inter-process communication to exchange
sharable handles, there is no difference in using fabric memory compared to
other allocation handle types.

**IMEX Channels Support**
Within an IMEX domain, IMEX channels enable secure memory sharing in
multi-user environments. The NVIDIA driver implements this by creating a
character device, `nvidia-caps-imex-channels`. To use fabric handle-based
sharing, users should verify two things:

- First, applications must verify that this device exists under
*/proc/devices*:

```c++
# cat /proc/devices | grep nvidia
195 nvidia
195 nvidiactl
234 nvidia-caps-imex-channels
509 nvidia-nvswitch

The nvidia-caps-imex-channels device should have a major number (e.g., 234).
```

- Second, for two CUDA processes (an exporter and an importer) to share memory,
they must both have access to the same IMEX channel file. These files, such
as */dev/nvidia-caps-imex-channels/channel0*, are nodes that represent
individual IMEX channels. System administrators must create these files, for
example, using the *mknod()* command.

```c++
# mknod /dev/nvidia-caps-imex-channels/channelN c <major_number> 0

This command creates channelN using the major number obtained from
/proc/devices.
```

> **Note**
>
> By default, the driver can create channel0
> if the *NVreg_CreateImexChannel0* module parameter is specified.

**Multicast Object Support:**
Before attempting to use multicast objects, applications must ensure that the
devices they want to use support them. The following code sample
shows querying for multicast object support:

```c++
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` supports Multicast Objects
}
```
