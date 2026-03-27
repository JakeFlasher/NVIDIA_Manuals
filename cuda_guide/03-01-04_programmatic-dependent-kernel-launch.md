---
title: "3.1.4. Programmatic Dependent Kernel Launch"
section: "3.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#programmatic-dependent-kernel-launch"
---

## [3.1.4. Programmatic Dependent Kernel Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#programmatic-dependent-kernel-launch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#programmatic-dependent-kernel-launch "Permalink to this headline")

As we have discussed earlier, the semantics of CUDA Streams are such that kernels execute in order. This is so that if we have two successive kernels, where the second kernel depends on results from the first one, the programmer can be safe in the knowledge that by the time the second kernel starts executing the dependent data will be available. However, it may be the case that the first kernel can have the data on which a subsequent kernel depends already written to global memory and it still has more work to do. Likewise the dependent second kernel may have some independent work before it needs the data from the first kernel. In such a situation it is possible to partially overlap the execution of the two kernels (assuming that hardware resources are available). The overlapping can also overlap the launch overheads of the second kernel too. Other than the availability of hardware resources,the degree of overlap which can be achieved is dependent on the specific structure of the kernels, such as

- when in its execution does the first kernel finish the work on which the second kernel depends?
- when in its execution does the second kernel start working on the data from the first kernel?

since this is very much dependent on the specific kernels in question it is difficult to automate completely and hence CUDA provides a mechanism to allow the application developer to specify the synchronization point between the two kernels. This is done via a technique known as Programmatic Dependent Kernel Launch. The situation is depicted in the figure below.

![Programmatic Dependent Kernel Launch](images/____________-_________-______-_______1.png)

PDL has three main components.

1. The first kernel (the so called _primary kernel_) needs to call a special function to indicate that it is done with the everything that the subsequent dependent kernels (also called _secondary kernel_) will need. This is done by calling the function  `cudaTriggerProgrammaticLaunchCompletion()`.
2. In turn, the dependent secondary kernel needs to indicate that it has reached the portion of the its work which is independent of the primary kernel and that it is now waiting on the primary kernel to finish the work on which it depends. This is done with the function `cudaGridDependencySynchronize()`.
3. THe second kernel needs to be launched with a special attribute *cudaLaunchAttributeProgrammaticStreamSerialization* with its *programmaticStreamSerializationAllowed* field set to ‘1’.

The following code snippet shows an example of how this can be done.

*Listing 3 Example of Programmatic Dependent Kernel Launch with two Kernels[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#pdl-example "Link to this code")*

```c
__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel

    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
    // Initialization, Independent work, etc.

    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();

    // Dependent work
}

// Launch the secondary kernel with the special attribute

// Set Up the attribute
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;

// Set the attribute in a kernel launch configuration
 cudaLaunchConfig_t config = {0};

// Base launch configuration
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes= 0;
config.stream = stream;

// Add special attribute for PDL
config.attrs = attribute;
config.numAttrs = 1;

// Launch primary kernel
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

// Launch secondary (dependent) kernel using the configuration with
// the attribute
cudaLaunchKernelEx(&config, secondary_kernel);
```
