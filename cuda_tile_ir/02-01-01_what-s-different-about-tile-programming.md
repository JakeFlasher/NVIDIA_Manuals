---
title: "2.1.1. What’s different about tile programming?"
section: "2.1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#what-s-different-about-tile-programming"
---

### [2.1.1. What’s different about tile programming?](https://docs.nvidia.com/cuda/tile-ir/latest/sections#what-s-different-about-tile-programming)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#what-s-different-about-tile-programming "Permalink to this headline")

**Tile IR** is an extension to the CUDA programming model that enables first class support for tile programming.
Tiled kernels express programs as a grid of logical tile threads that operate over tiles. The mapping of both
the grid and individual tile threads to the underlying hardware’s threads is abstracted away from the
programming model and is handled by the compiler.

The SIMT programming model of NVIDIA’s streaming multiprocessor (SM) is one in which threads operate over
(relatively) small pieces of data and the user is responsible for dividing and scheduling the threads into
the appropriate blocks to compute over the input data in an efficient manner. This model gives flexibility
to programmers on how to map threads to data, or vice-versa. SIMT is the programming model exposed by
CUDA and PTX and has served NVIDIA GPUs well since its introduction in 2006.

The rise in importance of deep learning has both introduced a greater regularity to user workloads
and an ever increasing need to deliver performance for these workloads. As discussed in the
[Introduction](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html#section-introduction), this has led to new specialized hardware in the form of tensor cores.

Tensor cores introduce a new dimension to the SIMT programming model. Now, SM threads must cooperate with
the tensor cores in order to reach peak performance. With each new generation of hardware the interplay between
these two pieces of silicon has unlocked amazing new performance but with increasing programming complexity.

**Tile IR** has been built to aid in the implementation of high-performance algorithms that take full advantage
of the underlying hardware’s capabilities while mitigating the increase in programming complexity.

By abstracting thread-to-data mapping, **Tile IR** simplifies the use of specialized hardware like Tensor Cores compared to
traditional SIMT models.
