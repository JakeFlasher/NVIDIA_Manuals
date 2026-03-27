---
title: "Device Allocations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#device-allocations"
---

## [Device Allocations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#device-allocations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#device-allocations "Permalink to this headline")

To strictly allocate memory on the device using the smart pointer pattern to manage allocation and deallocation,
use `cutlass::DeviceAllocation<>`.

**Example:** allocating an array in device memory.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor_view.h>
#include <cutlass/util/device_memory.h>

__global__ void kernel(float *device_ptr) {

}

int main() {

  size_t N = 1024;

  cutlass::DeviceAllocation<float> device_alloc(N);

  // Call a CUDA kernel passing device memory as a pointer argument
  kernel<<< grid, block >>>(alloc.get());

  if (cudaGetLastError() != cudaSuccess) {
    return -1;
  }

  // Device memory is automatically freed when device_alloc goes out of scope

  return 0;
}
```
