---
title: "3. The CUDA Compilation Trajectory"
section: "3"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory"
---

# [3. The CUDA Compilation Trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#the-cuda-compilation-trajectory)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory "Permalink to this headline")

CUDA compilation works as follows: the input program is preprocessed for device compilation and is compiled to CUDA binary (`cubin`) and/or PTX intermediate code, which are placed in a fatbinary. The input program is preprocessed once again for host compilation and is synthesized to embed the fatbinary and transform CUDA specific C++ extensions into standard C++ constructs. Then the C++ host compiler compiles the synthesized host code with the embedded fatbinary into a host object. The exact steps that are followed to achieve this are displayed in [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure).

The embedded fatbinary is inspected by the CUDA runtime system whenever the device code is launched by the host program to obtain an appropriate fatbinary image for the current GPU.

CUDA programs are compiled in the whole program compilation mode by default, i.e., the device code cannot reference an entity from a separate file. In the whole program compilation mode, device link steps have no effect. For more information on the separate compilation and the whole program compilation, refer to [Using Separate Compilation in CUDA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-separate-compilation-in-cuda).

![CUDA Compilation Trajectory](images/the-cuda-compilation-trajectory_1.png)

CUDA Compilation Trajectory
