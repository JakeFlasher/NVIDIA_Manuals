---
title: "1.3.4. Cubins and Fatbins"
section: "1.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cubins-and-fatbins"
---

## [1.3.4. Cubins and Fatbins](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#cubins-and-fatbins)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#cubins-and-fatbins "Permalink to this headline")

CUDA applications and libraries are usually written in a higher-level language like C++. That higher-level language is compiled to PTX, and then the PTX is compiled into real binary for a physical GPU, called a _CUDA binary_, or _cubin_ for short. A cubin has a specific binary format for a specific SM version, such as *sm_120*.

Executables and library binaries that use GPU computing contain both CPU and GPU code. The GPU code is stored within a container called a _fatbin_. Fatbins can contain cubins and PTX for multiple different targets. For example, an application could be built with binaries for multiple different GPU architectures, that is, different SM versions. When an application is run, its GPU code is loaded onto a specific GPU and the best binary for that GPU from the fatbin is used.

![Fatbin containers within executables or libraries can contain multiple GPU code versions](images/______-___-________1.png)

Figure 8 The binary for an executable or library contains both CPU binary code and a fatbin container for GPU code. A fatbin can contain both cubin GPU binary code and PTX virtual ISA code. PTX code can be JIT compiled for future targets.[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#fatbin-graphic "Link to this image")

Fatbins can also contain one or more PTX versions of GPU code, the use for which is described in [PTX Compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#cuda-platform-ptx-compatibility). [Figure 8](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#fatbin-graphic) shows an example of an application or library binary which contains multiple cubin versions of GPU code as well as one version of PTX code.
