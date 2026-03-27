---
title: "2.4.2.1.1. Unified Memory Paradigm: Code Example"
section: "2.4.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-memory-paradigm-code-example"
---

#### [2.4.2.1.1. Unified Memory Paradigm: Code Example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#unified-memory-paradigm-code-example)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#unified-memory-paradigm-code-example "Permalink to this headline")

The following code example demonstrates querying the device attributes and determining the unified memory paradigm, following the logic of [Figure 18](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#unified-memory-flow-chart), for each GPU in a system.

```cuda
void queryDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    for(int i=0; i<numDevices; i++)
    {
        cudaSetDevice(i);
        cudaInitDevice(0, 0, 0);
        int deviceId = i;

        int concurrentManagedAccess = -1;
        cudaDeviceGetAttribute (&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);
        int pageableMemoryAccess = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
        int pageableMemoryAccessUsesHostPageTables = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId);

        printf("Device %d has ", deviceId);
        if(concurrentManagedAccess){
            if(pageableMemoryAccess){
                printf("full unified memory support");
                if( pageableMemoryAccessUsesHostPageTables)
                    { printf(" with hardware coherency\n");  }
                else
                    { printf(" with software coherency\n"); }
            }
            else
                { printf("full unified memory support for CUDA-made managed allocations\n"); }
        }
        else
        {   printf("limited unified memory support: Windows, WSL, or Tegra\n");  }
    }
}
```
