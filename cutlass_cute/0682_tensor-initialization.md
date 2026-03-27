---
title: "Tensor Initialization"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#tensor-initialization"
---

## [Tensor Initialization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tensor-initialization)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tensor-initialization "Permalink to this headline")

CUTLASS defines several utility functions to initialize tensors to uniform, procedural,
or randomly generated elements. These have implementations using strictly host code and
implementations using strictly CUDA device code.

`TensorFill()` for uniform elements throughout a tensor.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  float x = 3.14159f;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFill(tensor.host_view(), x);

  // Initialize in device memory
  cutlass::reference::device::TensorFill(tensor.device_view(), x);

  return 0;
}
```

`TensorFillRandomUniform()` for initializing elements to a random uniform distribution.
The device-side implementation uses CURAND to generate random numbers.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  double maximum = 4;
  double minimum = -4;
  uint64_t seed = 0x2019;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomUniform(
    tensor.host_view(),
    seed,
    maximum,
    minimum);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomUniform(
    tensor.device_view(),
    seed,
    maximum,
    minimum);

  return 0;
}
```

`TensorFillRandomGaussian()` for initializing elements to a random gaussian distribution.
The device-side implementation uses CURAND to generate random numbers.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {

  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev);

  return 0;
}
```

Each of these functions accepts an additional argument to specify how many bits of
the mantissa less than 1 are non-zero. This simplifies functional comparisons when
exact random distributions are not necessary, since elements may be restricted to
integers or values with exact fixed-point representations.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {

  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  int bits_right_of_binary_decimal = 2;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev,
    bits_right_of_binary_decimal);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev,
    bits_right_of_binary_decimal);

  return 0;
}
```

These utilities may be used for all data types.

**Example:** random half-precision tensor with Gaussian distribution.

```c++
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  // Allocate a column-major tensor with half-precision elements
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev);

  return 0;
}
```
