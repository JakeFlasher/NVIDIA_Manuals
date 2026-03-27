---
title: "1.3.4.2. PTX Compatibility"
section: "1.3.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#ptx-compatibility"
---

### [1.3.4.2. PTX Compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#ptx-compatibility)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#ptx-compatibility "Permalink to this headline")

GPU code can be stored in executables in binary or PTX form, which is covered in [Cubins and Fatbins](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#cuda-platform-cubins-fatbins). When an application stores the PTX version of GPU code, that PTX can be JIT compiled at application runtime for any compute capability equal or higher to the compute capability of the PTX code. For example, if an application contains PTX for *compute_80*, that PTX code can be JIT compiled to later SM versions, such as *sm_120* at application runtime. This enables forward compatibility with future GPUs without the need to rebuild applications or libraries.
