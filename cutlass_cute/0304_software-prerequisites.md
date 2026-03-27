---
title: "Software prerequisites"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html#software-prerequisites"
---

## [Software prerequisites](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#software-prerequisites)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#software-prerequisites "Permalink to this headline")

1. Windows 10 or 11
2. Visual Studio 2019 version 16.11.27, or Visual Studio 2022
3. CUDA Toolkit (at least 12.2; earlier 12.x versions may work)
4. CMake (at least 3.18)
5. git
6. Python (at least 3.6)

Visual Studio must be installed _before_ the CUDA Toolkit.
Otherwise, Visual Studio’s build system won’t know about CUDA.
