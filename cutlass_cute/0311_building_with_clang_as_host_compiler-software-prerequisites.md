---
title: "Software prerequisites"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_with_clang_as_host_compiler.html#building_with_clang_as_host_compiler--software-prerequisites"
---

## [Software prerequisites](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#software-prerequisites)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#software-prerequisites "Permalink to this headline")

1. Clang (regularly tested with Clang 17;
occasionally tested with Clang 10 and greater)
2. CUDA Toolkit (tested with 12.2; other versions likely work)
3. CMake (at least 3.18)
4. git
5. Python (at least 3.6)

Experience with Ubuntu 22.04 LTS is that
clang requires the following packages to be installed.

```bash
$ sudo apt-get install clang cmake ninja-build pkg-config libgtk-3-dev liblzma-dev libstdc++-12-dev
```

A symptom of not installing all needed dependencies
is the following error when attempting to use clang:
`"/usr/bin/ld: cannot find -lstdc++: No such file or directory"`.
