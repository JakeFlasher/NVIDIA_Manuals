---
title: "4.6.4.3. Step 2 (continued): Add workqueue resources"
section: "4.6.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#step-2-continued-add-workqueue-resources"
---

### [4.6.4.3. Step 2 (continued): Add workqueue resources](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#step-2-continued-add-workqueue-resources)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#step-2-continued-add-workqueue-resources "Permalink to this headline")

If you also want to specify a workqueue resource, then this needs to be done explicitly.
The following example shows how to create a workqueue configuration resource for a specific device with balanced sharing scope and a concurrency limit of four.

```c++
cudaDevResource split_result[2] = {{}, {}};
// code to populate split_result[0] not shown; used split API with nbGroups=1

// The last resource will be a workqueue resource.
split_result[1].type = cudaDevResourceTypeWorkqueueConfig;
split_result[1].wqConfig.device = 0; // assume device ordinal of 0
split_result[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
split_result[1].wqConfig.wqConcurrencyLimit = 4;
```

A workqueue concurrency limit of four hints to the driver that the user expects maximum four concurrent stream-ordered workloads.
The driver will assign work queues trying to respect this hint, if possible.
