---
title: "Example CMake Commands"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#example-cmake-commands"
---

## [Example CMake Commands](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#example-cmake-commands)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-cmake-commands "Permalink to this headline")

To instantiate all operations supporting all tile sizes, data types, and alignment constraints, specify
`-DCUTLASS_LIBRARY_KERNELS=all` when running `cmake`.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='70;75;80' -DCUTLASS_LIBRARY_KERNELS=all
```

The above command line generates about twenty thousand kernels targeting NVIDIA Ampere, Turing, and Volta architectures.
Compiling thousands of kernels for three different architectures is time-consuming. Additionally, this would also result
in a large binary size and on some platforms linker to fail on building the library.

Enabling the “unity build” instantiates multiple kernel instances in each compilation unit, thereby reducing binary size
and avoiding linker limitations on some platforms.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

It is advised to only compile CUTLASS kernels for NVIDIA architectures one plans on running. Furthermore, kernels
can be selectively included in the CUTLASS Library by specifying filter strings and wildcard characters when executing CMake.

Several examples are defined below for convenience. They may be combined as a comma-delimited list.
Compling only the kernels desired reduces compilation time.
