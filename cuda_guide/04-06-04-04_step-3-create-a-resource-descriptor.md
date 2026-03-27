---
title: "4.6.4.4. Step 3: Create a Resource Descriptor"
section: "4.6.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#step-3-create-a-resource-descriptor"
---

### [4.6.4.4. Step 3: Create a Resource Descriptor](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#step-3-create-a-resource-descriptor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#step-3-create-a-resource-descriptor "Permalink to this headline")

The next step, after resources have been split, is to generate a resource descriptor, using the `cudaDevResourceGenerateDesc` API,
for all the resources expected to be available to a green context.

The relevant CUDA runtime API function signature is:

`cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t *phDesc, cudaDevResource *resources, unsigned int nbResources)`

It is possible to combine multiple `cudaDevResource` resources. For example, the code snippet below shows how to generate a resource descriptor that encapsulates three groups of resources.
You just need to ensure that these resources are all allocated continuously in the `resources` array.

```c++
cudaDevResource actual_split_result[5] = {};
// code to populate actual_split_result not shown

// Generate resource desc. to encapsulate 3 resources: actual_split_result[2] to [4]
cudaDevResourceDesc_t resource_desc;
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, &actual_split_result[2], 3));
```

Combining different types of resources is also supported. For example, one could generate a descriptor with both SM and workqueue resources.

For a `cudaDevResourceGenerateDesc` call to be successful:

- All `nbResources` resources should belong to the same GPU device.
- If multiple SM-type resources are combined, they should be generated from the same split API call and have the same `coscheduledSmCount` values (if not part of remainder)
- Only a single workqueue config or workqueue type resource may be present.
