---
title: "5.7. CUDA C++ Memory Model"
section: "5.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#cuda-c-memory-model"
---

# [5.7. CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-c-memory-model)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-c-memory-model "Permalink to this headline")

Standard C++ presents a view that the cost to synchronize threads is uniform and low.

CUDA C++ is different: the cost to synchronize threads grows as threads are further apart. It is low across threads
within a block, but high across arbitrary threads in the system running on multiple GPUs and CPUs.

To account for non-uniform thread synchronization costs that are not always low, CUDA C++ extends the standard C++
memory model and concurrency facilities in the `cuda::` namespace with **thread scopes**, retaining the syntax and
semantics of standard C++ by default.
