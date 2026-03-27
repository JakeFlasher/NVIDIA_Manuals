---
title: "Building"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html#building"
---

## [Building](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#building)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#building "Permalink to this headline")

A successful CMake run will create a `CUTLASS.sln` Visual Studio “solution” file in the build directory.
One can open this in Visual Studio and build the entire solution or any subset of projects as desired.
It may be necessary to limit maximum build parallelism by setting the appropriate Visual Studio option.

Alternately, one can run `cmake --build . --config Release -j 4` in the build directory.
Replace 4 with the desired maximum build parallelism.
It’s important to put the `--build` option before the period that signifies the build directory.
The `--config` option specifies the kind of build;
`--config Release` builds a Release build, while `--config Debug` builds a Debug build.
Unlike with CMake’s Makefile or Ninja generators,
`CMAKE_BUILD_TYPE` has no effect on the Visual Studio generator,
because the Visual Studio generator creates all build configurations.
