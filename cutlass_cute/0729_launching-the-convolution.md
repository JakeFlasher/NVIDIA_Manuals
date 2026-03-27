---
title: "Launching the convolution"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#launching-the-convolution"
---

## [Launching the convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#launching-the-convolution)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#launching-the-convolution "Permalink to this headline")

The following code collects the arguments for an implicit GEMM operation into a structure.

```c++
//
// Define arguments for CUTLASS Convolution
//

// mode (kCrossCorrelation or kConvolution)
cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

// Split K dimension into 1 partitions
int split_k_slices = 1;

cutlass::conv::Conv2dProblemSize problem_size(
    options.input_size,
    options.filter_size,
    options.padding,
    options.conv_stride,
    options.dilation,
    options.output_size(),
    mode,
    split_k_slices);

typename ImplicitGemm::Arguments arguments{
  problem_size,
  tensor_a.device_ref(),
  tensor_b.device_ref(),
  tensor_c.device_ref(),
  tensor_c.device_ref(),
  {options.alpha, options.beta},
};
```

The `mode` flag indicates whether to compute cross correlation or convolution. The arguments
`input_size`, `filter_size`, `padding`, `conv_stride`, and `dilation` specify the dimensions of the
input and output tensors and characterize the problem size.

The arguments `tensor_a.device_ref()`, `tensor_b.device_ref()`, and `tensor_c.device_ref()` are
CUTLASS `TensorRef<>` objects containing a pointer to the tensor data in GPU device memory and stride values.

The following code initializes and launches the Implicit GEMM operation on the device. After initializing
the arguments structure, it is used to query device-side workspace requirements and allocate them
in device memory if needed.

Then, the Implicit GEMM object is initialized with the `arguments` structure and the workspace in
device memory. This initialization step precomputes internal lookup tables used by the convolution kernel
and may also clear the device-side workspace if needed.

Finally, the initialized Implicit GEMM object is called, launching a kernel on the device. `tensor_c` now
contains the result of the implicit GEMM.

```c++
ImplicitGemm implicit_gemm_op;

// Query workspace size
size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

// Allocate workspace memory
cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

// Initialize the Implicit GEMM object
cutlass::Status status = implicit_gemm_op.initialize(arguments, workspace.get());

if (status != cutlass::Status::kSuccess) {
  /* error */
}

//
// Launch initialized CUTLASS kernel
//

status = implicit_gemm_op();

if (status != cutlass::Status::kSuccess) {
  /* error */
}
```

The example demonstrates how the input and output tensors may be written to a file as CSV using
`cutlass::HostTensor<>` defined in the [CUTLASS Utilities](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html).

```c++
  std::ofstream output_workspace(ss.str());

  output_workspace
    << "Input = \n" << tensor_a.host_view() << "\n\n"
    << "Filters = \n" << tensor_b.host_view() << "\n\n";

  // Copy device memory to host backing store
  tensor_c.sync_host();

  output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;
```
