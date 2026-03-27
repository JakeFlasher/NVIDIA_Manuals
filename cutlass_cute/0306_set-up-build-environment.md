---
title: "Set up build environment"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html#set-up-build-environment"
---

## [Set up build environment](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#set-up-build-environment)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#set-up-build-environment "Permalink to this headline")

1. Run “git bash” to get a familiar command-line interface
2. Edit `~/.profile` and set the environment variables as needed to access the CUTLASS repository
3. Clone the CUTLASS repository
4. Create the `build` subdirectory in the CUTLASS clone directory, and run CMake in it,
specifying whatever CMake options are desired, e.g.,
`cmake .. -DCUTLASS_NVCC_ARCHS=90a`

Alternate approaches may rely on the CMake GUI and/or Windows’ native command line.
