---
title: "CUDA Programming Guide"
section: ""
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#cuda-programming-guide"
---

# [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide#cuda-programming-guide)[](https://docs.nvidia.com/cuda/cuda-programming-guide/#cuda-programming-guide "Permalink to this headline")

**CUDA and the CUDA Programming Guide**

CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. It allows developers to accelerate compute-intensive applications and is widely used in fields such as deep learning, scientific computing, and high-performance computing (HPC).

This CUDA Programming Guide is the official, comprehensive resource on the CUDA programming model and how to write code that executes on the GPU using the CUDA platform. This guide covers everything from the CUDA programming model and the CUDA platform to the details of language extensions and covers how to make use of specific hardware and software features. This guide provides a pathway for developers to learn CUDA if they are new, and also provides an essential resource for developers as they build applications using CUDA.

**Organization of This Guide**

Even for developers who primarily use libraries, frameworks, or DSLs, an understanding of the CUDA programming model and how GPUs execute code is valuable in knowing what is happening behind the layers of abstraction.
This guide starts with a chapter on the [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model) outside of any specific programming language which is applicable to anyone interested in understanding how CUDA works, even non-developers.

The guide is broken down into five primary parts:

- Part 1: Introduction and Programming Model Abstract
  - A language agnostic overview of the CUDA programming model as well as a brief tour of the CUDA platform.
  - This section is meant to be read by anyone wanting to understand GPUs and the concepts of executing code on GPUs, even if they are not developers.
- Part 2: Programming GPUs in CUDA
  - The basics of programming GPUs using CUDA C++.
  - This section is meant to be read by anyone wanting to get started in GPU programming.
  - This section is meant to be instructional, not complete, and teaches the most important and common parts of CUDA programming, including some common performance considerations.
- Part 3: Advanced CUDA
  - Introduces some more advance features of CUDA that enable both fine-grained control and more opportunities to maximize performance, including the use of multiple GPUs in a single application.
  - This section concludes with a [tour of the features covered in part 4](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#tour-of-features) with a brief introduction to the purpose and function of each, sorted by when and why a developer may find each feature useful.
- Part 4: CUDA Features
  - This section contains complete coverage of specific CUDA features such as CUDA graphs, dynamic parallelism, interoperability with graphics APIs, and unified memory.
  - This section should be consulted when knowing the complete picture of a specific CUDA feature is needed. Where possible, care has been taken to introduce and motivate the features covered in this section in earlier sections.
- Part 5: Technical Appendices
  - The technical appendices provide some reference documentation on CUDA’s C++ high-level language support, hardware-specific specifications, and other technical specifications.
  - This section is meant as technical reference for specific description of syntax, semantics, and technical behavior of elements of CUDA.

Parts 1-3 provide a guided learning experience for developers new to CUDA, though they also provide insight and updated information useful for CUDA developers of any experience level.

Parts 4 and 5 provide a wealth of information about specific features and detailed topics, and are intended to provide a curated, well-organized reference for developers needing to know more details as they write CUDA applications.
