---
title: "CUTLASS is a C++ project"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#cutlass-is-a-c-project"
---

#### [CUTLASS is a C++ project](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-is-a-c-project)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-is-a-c-project "Permalink to this headline")

CUTLASS is a C++ project.  CUDA C++ is a C++ dialect.
Therefore, we write using standard C++ idioms as much as possible.
We aim for portability to as many compilers as possible,
by writing host code in Standard C++
and device code in CUDA C++
that resembles Standard C++ as much as possible.
This improves usability
for the general community of C++ developers,
and makes it easier for new staff to join the project.
