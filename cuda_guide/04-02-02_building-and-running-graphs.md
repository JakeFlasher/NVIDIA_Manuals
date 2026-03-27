---
title: "4.2.2. Building and Running Graphs"
section: "4.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#building-and-running-graphs"
---

## [4.2.2. Building and Running Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#building-and-running-graphs)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#building-and-running-graphs "Permalink to this headline")

Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution.

- During the **definition** or **creation** phase, a program creates a description of the operations in the graph along with the dependencies between them.
- **Instantiation** takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an _executable graph._
- An **executable** graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.
