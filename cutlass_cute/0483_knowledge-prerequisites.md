---
title: "Knowledge prerequisites"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html#knowledge-prerequisites"
---

## [Knowledge prerequisites](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#knowledge-prerequisites)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#knowledge-prerequisites "Permalink to this headline")

CuTe is a CUDA C++ header-only library.  It requires C++17
(the revision of the C++ Standard that was released in 2017).

Throughout this tutorial, we assume intermediate C++ experience.
For example, we assume that readers know
how to read and write templated functions and classes, and
how to use the `auto` keyword to deduce a function’s return type.
We will be gentle with C++ and explain some things
that you might already know.

We also assume intermediate CUDA experience.
For example, readers must know
the difference between device and host code,
and how to launch kernels.
