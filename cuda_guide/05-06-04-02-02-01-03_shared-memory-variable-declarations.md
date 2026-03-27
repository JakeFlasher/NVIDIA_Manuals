---
title: "5.6.4.2.2.1.3. Shared Memory Variable Declarations"
section: "5.6.4.2.2.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#shared-memory-variable-declarations"
---

###### [5.6.4.2.2.1.3. Shared Memory Variable Declarations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#shared-memory-variable-declarations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#shared-memory-variable-declarations "Permalink to this headline")

In CUDA C++ shared memory can be declared either as a statically sized file-scope or function-scoped variable, or as an `extern` variable with the size determined at runtime by the kernel’s caller via a launch configuration argument. Both types of declarations are valid under the device runtime.

```c++
__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```
