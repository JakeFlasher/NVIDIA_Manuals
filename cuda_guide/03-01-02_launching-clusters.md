---
title: "3.1.2. Launching Clusters:"
section: "3.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#launching-clusters"
---

## [3.1.2. Launching Clusters:](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#launching-clusters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#launching-clusters "Permalink to this headline")

[Thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-thread-block-clusters), introduced in previous sections, are an optional level of thread block organization available in compute capability 9.0 and higher which enable applications to guarantee that thread blocks of a cluster are simultaneously executed on single GPC. This enables larger groups of threads than those that fit in a single SM to exchange data and synchronize with each other.

Section [Section 2.1.10.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-cluster-triple-chevron) showed how a kernel which uses clusters can be specified and launched using triple chevron notation. In this section, the `__cluster_dims__` annotation was used to specify the dimensions of the cluster which must be used to launch the kernel. When using triple chevron notation, the size of the clusters is determined implicitly.
