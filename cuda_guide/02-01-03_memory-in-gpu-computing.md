---
title: "2.1.3. Memory in GPU Computing"
section: "2.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#memory-in-gpu-computing"
---

## [2.1.3. Memory in GPU Computing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#memory-in-gpu-computing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-in-gpu-computing "Permalink to this headline")

In order to use the `vecAdd` kernel shown above, the arrays `A`, `B`, and `C` must be in memory accessible to the GPU. There are several different ways to do this, two of which will be illustrated here. Other methods will be covered in later sections on [unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory). The memory spaces available to code running on the GPU were introduced in [GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-memory) and are covered in more detail in [GPU Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-gpu-device-memory-spaces).
