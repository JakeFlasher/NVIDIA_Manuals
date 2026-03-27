---
title: "Required CMake options"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_with_clang_as_host_compiler.html#required-cmake-options"
---

### [Required CMake options](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#required-cmake-options)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#required-cmake-options "Permalink to this headline")

The Clang build requires specifying the following CMake options.
Replace `<path-to-clang++>` with the path to your `clang++` executable.
You may use `clang++` directly if it is in your `PATH`.

- `CMAKE_CXX_COMPILER=<path-to-clang++>`
- `CMAKE_CUDA_HOST_COMPILER=<path-to-clang++>`

One must set both!  It’s not enough just to set the `CXX` environment
variable, for example.  Symptoms of only setting `CMAKE_CXX_COMPILER`
(or only setting the `CXX` environment variable) include `cc1plus`
(GCC’s compiler executable) reporting build errors due to it not
understanding Clang’s command-line options.

Users can also specify a particular CUDA Toolkit version
by setting the CMake option `CMAKE_CUDA_COMPILER`
to the path to the `nvcc` executable
that lives in the CUDA Toolkit’s directory.  For example,
if `${PATH_TO_CUDA_TOOLKIT}` is the CUDA Toolkit directory,
then one can set `CMAKE_CUDA_COMPILER` as follows.

- `CMAKE_CUDA_COMPILER=${PATH_TO_CUDA_TOOLKIT}/bin/nvcc`
