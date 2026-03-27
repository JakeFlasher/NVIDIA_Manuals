---
title: "Debugging Asynchronous Kernels with CUTLASS’s Built-in synclog Tool"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#debugging-asynchronous-kernels-with-cutlass-s-built-in-synclog-tool"
---

## [Debugging Asynchronous Kernels with CUTLASS’s Built-in synclog Tool](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#debugging-asynchronous-kernels-with-cutlass-s-built-in-synclog-tool)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#debugging-asynchronous-kernels-with-cutlass-s-built-in-synclog-tool "Permalink to this headline")

CUTLASS provides a built-in tool called `synclog` that enables printing runtime information useful for debugging asynchronous CUTLASS kernels. With the introduction of Warp Specialization in CUTLASS 3.0 for Hopper GPUs, kernel designs now incorporate synchronization among warps. The `synclog` tool simplifies debugging efforts for these asynchronous programs by recording and displaying timing information for synchronization events.
