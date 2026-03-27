---
title: "4.4.4.1. Avoiding Group Creation Hazards"
section: "4.4.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#avoiding-group-creation-hazards"
---

### [4.4.4.1. Avoiding Group Creation Hazards](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#avoiding-group-creation-hazards)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#avoiding-group-creation-hazards "Permalink to this headline")

Partitioning a group is a collective operation and all threads in the group must participate.
If the group was created in a conditional branch that not all threads reach, this can lead to deadlocks or data corruption.
