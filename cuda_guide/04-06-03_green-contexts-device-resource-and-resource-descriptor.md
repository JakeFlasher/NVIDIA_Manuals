---
title: "4.6.3. Green Contexts: Device Resource and Resource Descriptor"
section: "4.6.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts-device-resource-and-resource-descriptor"
---

## [4.6.3. Green Contexts: Device Resource and Resource Descriptor](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#green-contexts-device-resource-and-resource-descriptor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-device-resource-and-resource-descriptor "Permalink to this headline")

At the heart of a green context is a device resource (`cudaDevResource`) tied to a specific GPU device.
Resources can be combined and encapsulated into a descriptor (`cudaDevResourceDesc_t`).
A green context only has access to the resources encapsulated into the descriptor used for its creation.

Currently the `cudaDevResource` data structure is defined as:

```c++
struct {
     enum cudaDevResourceType type;
     union {
         struct cudaDevSmResource sm;
         struct cudaDevWorkqueueConfigResource wqConfig;
         struct cudaDevWorkqueueResource wq;
     };
 };
```

The supported valid resource types are `cudaDevResourceTypeSm`, `cudaDevResourceTypeWorkqueueConfig` and `cudaDevResourceTypeWorkqueue`, while  `cudaDevResourceTypeInvalid` identifies an invalid resource type.

A valid device resource can be associated with:

- a specific set of streaming multiprocessors (SMs) (resource type `cudaDevResourceTypeSm`),
- a specific workqueue configuration (resource type `cudaDevResourceTypeWorkqueueConfig`) or
- a pre-existing workqueue resource (resource type `cudaDevResourceTypeWorkqueue`).

One can query if a given execution context or CUDA stream is associated with a `cudaDevResource` resource of a given type,
using the `cudaExecutionCtxGetDevResource` and `cudaStreamGetDevResource` APIs respectively.
Being associated with different types of device resources (e.g., SMs and work queues) is also possible for an execution context,
while a stream can only be associated with an SM-type resource.

A given GPU device has, by default, all three device resource types: an SM-type resource encompassing all the SMs of the GPU, a workqueue configuration resource encompassing all available work queues
and its corresponding workqueue resource. These resources can be retrieved via the `cudaDeviceGetDevResource` API.

**Overview of relevant device resource structs**

The different resource type structs have fields that are set either explicitly by the user or by a relevant CUDA API call.
It is recommended to zero-initialize all device resource structs.

- An SM-type device resource (`cudaDevSmResource`) has the following relevant fields:

The above fields will be set via either the appropriate split API (`cudaDevSmResourceSplitByCount` or `cudaDevSmResourceSplit`) used to create this SM-type resource or will be populated by the `cudaDeviceGetDevResource` API which retrieves the SM resources of a given GPU device. These fields should never be set directly by the user. See next section for more details.
  - `unsigned int smCount`: number of SMs available in this resource
  - `unsigned int minSmPartitionSize`: minimum SM count required to partition this resource
  - `unsigned int smCoscheduledAlignment`: number of SMs in the resource guaranteed to be co-scheduled on the same GPU processing cluster, which is relevant for thread block clusters. `smCount` is a multiple of this value when  `flags` is zero.
  - `unsigned int flags`: supported flags are  0 (default) and `cudaDevSmResourceGroupBackfill` (see `cudaDevSmResourceGroup` flags).
- A workqueue configuration device resource (`cudaDevWorkqueueConfigResource`) has the following relevant fields:

These fields need to be set by the user. There is no CUDA API similar to the split APIs that generates a workqueue configuration resource, with the exception of the workqueue configuration resource populated by the `cudaDeviceGetDevResource` API. That API can retrieve the workqueue configuration resources of a given GPU device.
  - `int device`: the device on which the workqueue resources are available
  - `unsigned int wqConcurrencyLimit`: the number of stream-ordered workloads expected to avoid false dependencies
  - `enum cudaDevWorkqueueConfigScope sharingScope`: the sharing scope for the workqueue resources. Supported values are: `cudaDevWorkqueueConfigScopeDeviceCtx` (default) and `cudaDevWorkqueueConfigScopeGreenCtxBalanced`. With the default option, all workqueue resources are shared across all contexts, while with the balanced option the driver tries to use non-overlapping workqueue resources across green contexts wherever possible, using the user-specified `wqConcurrencyLimit` as a hint.
- Finally, a pre-existing workqueue resource (`cudaDevResourceTypeWorkqueue`) has no fields that can be set by the user. As with the other resource types, `cudaDevGetDevResource` can retrieve the pre-existing workqueue resource of a given GPU device.
