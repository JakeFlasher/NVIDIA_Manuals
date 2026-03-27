---
title: "4.4.3.1. Create Implicit Group Handles As Early As Possible"
section: "4.4.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#create-implicit-group-handles-as-early-as-possible"
---

### [4.4.3.1. Create Implicit Group Handles As Early As Possible](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#create-implicit-group-handles-as-early-as-possible)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#create-implicit-group-handles-as-early-as-possible "Permalink to this headline")

For best performance it is recommended that you create a handle for the implicit group upfront (as early as possible, before any branching has occurred) and use that handle throughout the kernel.
