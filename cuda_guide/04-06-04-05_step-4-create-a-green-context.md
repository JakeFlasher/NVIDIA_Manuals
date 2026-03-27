---
title: "4.6.4.5. Step 4: Create a Green Context"
section: "4.6.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#step-4-create-a-green-context"
---

### [4.6.4.5. Step 4: Create a Green Context](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#step-4-create-a-green-context)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#step-4-create-a-green-context "Permalink to this headline")

The final step is to create a green context from a resource descriptor using the `cudaGreenCtxCreate` API.
That green context will only have access to the resources (e.g., SMs, work queues) encapsulated in the resource descriptor specified during its creation.
These resources will be provisioned during this step.

The relevant CUDA runtime API function signature is:

`cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t *phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags)`

The `flags` parameter should be set to 0.
It is also recommended to explicitly initialize the device’s primary context before creating a green context
via either the `cudaInitDevice` API or the `cudaSetDevice` API, which also sets the primary context as current to the calling thread.
Doing so ensures there will be no additional primary context initialization overhead during green context creation.

See code snippet below.

```c++
int current_device = 0; // assume single GPU
CUDA_CHECK(cudaSetDevice(current_device)); // Or cudaInitDevice

cudaDevResourceDesc_t resource_desc {};
// Code to generate resource_desc not shown

// Create a green_ctx on GPU with current_device ID with access to resources from resource_desc
cudaExecutionContext_t green_ctx {};
CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, current_device, 0));
```

After a successful green context creation, the user can verify its resources by calling `cudaExecutionCtxGetDevResource` on that
execution context for each resource type.

**Creating Multiple Green Contexts**

An application can have more than one green context, in which case some of the steps above should be repeated.
For most use cases, these green contexts will each have a separate non-overlapping set of provisioned SMs.
For example, for the case of five homogeneous `cudaDevResource` groups (`actual_split_result` array), one green context’s descriptor may encapsulate
actual_split_result[2] to [4] resources, while the descriptor of another green context may encapsulate actual_split_result[0] to [1].
In this case, a specific SM will be provisioned for only one of the two green contexts of the application.

But SM oversubscription is also possible and may be used in some cases.
For example, it may be acceptable to have the second green context’s descriptor encapsulate actual_split_result[0] to [2].
In this case, all the SMs of actual_split_resource[2] `cudaDevResource` will be oversubscribed, i.e., provisioned for both green contexts,
while resources actual_split_resource[0] to [1] and actual_split_resource[3] to [4] may only be used by one of the two green contexts.
SM oversubscription should be judiciously used on a per-case basis.
