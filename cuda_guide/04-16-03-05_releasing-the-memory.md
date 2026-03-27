---
title: "4.16.3.5. Releasing the Memory"
section: "4.16.3.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#releasing-the-memory"
---

### [4.16.3.5. Releasing the Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#releasing-the-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#releasing-the-memory "Permalink to this headline")

To release the allocated memory and address space, both the source and target
processes should use *cuMemUnmap*, *cuMemRelease*, and *cuMemAddressFree* functions
in that order. The *cuMemUnmap* function un-maps a previously mapped memory
region from an address range, effectively detaching the physical memory from
the reserved virtual address space. Next, *cuMemRelease* deallocates the
physical memory that was previously created, returning it to the system.
Finally, *cuMemAddressFree* frees a virtual address range that was previously
reserved, making it available for future use. This specific order ensures a
clean and complete deallocation of both the physical memory and the virtual
address space.

```c++
cuMemUnmap(ptr, size);
cuMemRelease(handle);
cuMemAddressFree(ptr, size);
```

> **Note**
>
> In the OS-specific case, the exported handle must be closed using *fclose*.
> This step is not applicable to the fabric-based case.
