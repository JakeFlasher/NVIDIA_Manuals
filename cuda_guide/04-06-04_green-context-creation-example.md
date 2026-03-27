---
title: "4.6.4. Green Context Creation Example"
section: "4.6.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-context-creation-example"
---

## [4.6.4. Green Context Creation Example](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#green-context-creation-example)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-context-creation-example "Permalink to this headline")

There are four main steps involved in green context creation:

- Step 1: Start with an initial set of resources, e.g., by fetching the available resources of the GPU
- Step 2: Partition the SM resources into one or more partitions (using one of the available split APIs).
- Step 3: Create a resource descriptor combining, if needed, different resources
- Step 4: Create a green context from the descriptor, provisioning its resources

After the green context has been created, you can create CUDA streams belonging to that green context.
GPU work subsequently launched on such a stream, such as a kernel launched via `<<< >>>`, will only have access to this green context’s provisioned resources.
Libraries can also easily leverage green contexts, as long as the user passes a stream belonging to a green context to them.
See [Green Contexts - Launching work](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-launching-work) for more details.
