---
title: "4.5.2. API Description"
section: "4.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#api-description"
---

## [4.5.2. API Description](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#api-description)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#api-description "Permalink to this headline")

In Programmatic Dependent Launch, a primary and a secondary kernel are launched in the same CUDA stream.
The primary kernel should execute `cudaTriggerProgrammaticLaunchCompletion` with all thread blocks when
it’s ready for the secondary kernel to launch. The secondary kernel must be launched using the extensible launch API as shown.

```c++
__global__ void primary_kernel() {
   // Initial work that should finish before starting secondary kernel

   // Trigger the secondary kernel
   cudaTriggerProgrammaticLaunchCompletion();

   // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
   // Independent work

   // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
   cudaGridDependencySynchronize();

   // Dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

When the secondary kernel is launched using the `cudaLaunchAttributeProgrammaticStreamSerialization` attribute,
the CUDA driver is safe to launch the secondary kernel early and not wait on the
completion and memory flush of the primary before launching the secondary.

The CUDA driver can launch the secondary kernel when all primary thread blocks have launched and executed
`cudaTriggerProgrammaticLaunchCompletion`.
If the primary kernel doesn’t execute the trigger, it implicitly occurs after
all thread blocks in the primary kernel exit.

In either case, the secondary thread blocks might launch
before data written by the primary kernel is visible. As such, when the secondary kernel is configured with _Programmatic Dependent Launch_,
it must always use `cudaGridDependencySynchronize`
or other means to verify that the result data from the primary is available.

Please note that these methods provide the opportunity for the primary and secondary kernels to execute concurrently, however
this behavior is opportunistic and not guaranteed to lead to concurrent kernel execution.
Reliance on concurrent execution in this manner is unsafe and can lead to deadlock.
