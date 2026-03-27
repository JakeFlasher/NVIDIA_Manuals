---
title: "Tensor Allocation and I/O"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#tensor-allocation-and-i-o"
---

## [Tensor Allocation and I/O](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tensor-allocation-and-i-o)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tensor-allocation-and-i-o "Permalink to this headline")

To allocate a tensor with storage in both host and device memory, use `HostTensor` in
[`cutlass/util/host_tensor.h`](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/host_tensor.h)

```c++
template <typename Element, typename Layout>
class HostTensor;
```

This class is compatible with all CUTLASS numeric data types and layouts.

**Example:** column-major matrix storage of single-precision elements.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 32;
  int columns = 16;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  return 0;
}
```

Internal host-side storage may be accessed via the following methods.

```c++
float *host_ptr = tensor.host_data();
cutlass::TensorRef<float, cutlass::layout::ColumnMajor> host_ref = tensor.host_ref();
cutlass::TensorView<float, cutlass::layout::ColumnMajor> host_view = tensor.host_view();
```

Device memory may be accessed similarly.

```c++
float *device_ptr = tensor.device_data();
cutlass::TensorRef<float, cutlass::layout::ColumnMajor> device_ref = tensor.device_ref();
cutlass::TensorView<float, cutlass::layout::ColumnMajor> device_view = tensor.device_view();
```

Printing to human-readable CSV output is accoplished with `std::ostream::operator<<()` defined in
[`cutlass/util/tensor_view_io.h`](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/tensor_view_io.h).
Note, this assumes all views refer to host memory.

```c++
#include <cutlass/util/tensor_view_io.h>

int main() {
  // Obtain a TensorView into host memory
  cutlass::TensorView<float, cutlass::layout::ColumnMajor> view = tensor.host_view();

  // Print to std::cout
  std::cout << view << std::endl;

  return 0;
}
```

Host and device memory must be explicitly synchronized by the application.

```c++
float idx = 0;

for (int i = 0; i < rows; ++i) {
  for (int j = 0; j < columns; ++j) {

    // Write the element at location {i, j} in host memory
    tensor.host_ref().at({i, j}) = idx;

    idx += 0.5f;
  }
}

// Copy host memory to device memory
tensor.sync_device();

// Obtain a device pointer usable in CUDA kernels
float *device_ptr = tensor.device_data();
```

`HostTensor<>` is usable by all CUTLASS layouts including interleaved layouts.

```c++
int rows = 4;
int columns = 3;

cutlass::HostTensor<float, cutlass::layout::ColumnMajorInterleaved<4>> tensor({rows, columns});

for (int i = 0; i < rows; ++i) {
  for (int j = 0; j < columns; ++j) {

    // Write the element at location {i, j} in host memory
    tensor.host_ref().at({i, j}) = float(i) * 1.5f - float(j) * 2.25f;
  }
}

std::cout << tensor.host_view() << std::endl;
```
