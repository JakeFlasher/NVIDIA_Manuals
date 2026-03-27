---
title: "5.5.1. Floating-Point Introduction"
section: "5.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-introduction"
---

## [5.5.1. Floating-Point Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#floating-point-introduction)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#floating-point-introduction "Permalink to this headline")

Since the adoption of the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) for Binary Floating-Point Arithmetic in 1985, virtually all mainstream computing systems, including NVIDIA’s CUDA architectures, have implemented the standard. The IEEE-754 standard specifies how the results of floating-point arithmetic should be approximated.

To get accurate results and achieve the highest performance with the required precision, it is important to consider many aspects of floating-point behavior. This is particularly important in a heterogeneous computing environment where operations are performed on different types of hardware.

The following sections review the basic properties of floating-point computation and cover Fused Multiply-Add (FMA) operations and the dot product. These examples illustrate how different implementation choices affect accuracy.
