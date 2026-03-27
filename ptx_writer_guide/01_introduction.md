---
title: "1. Introduction"
section: "1"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#introduction"
---

# [1. Introduction](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#introduction)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#introduction "Permalink to this headline")

This document defines the Application Binary Interface (ABI) for the CUDA<sup>®</sup> architecture when generating PTX. By following the ABI, external developers can generate compliant PTX code that can be linked with other code.

PTX is a low-level parallel-thread-execution virtual machine and ISA (Instruction Set Architecture). PTX can be output from multiple tools or written directly by developers. PTX is meant to be GPU-architecture independent, so that the same code can be reused for different GPU architectures. For more information on PTX, refer to the latest version of the [PTX ISA reference document](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

There are multiple CUDA architecture families, each with their own ISA; e.g. SM 5.x is the Maxwell family, SM 6.x is the Pascal family. This document describes the high-level ABI for all architectures. Programs conforming to an ABI are expected to be executed on the appropriate architecture GPU, and can assume that instructions from that ISA are available.
