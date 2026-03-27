---
title: "3.5.4.2. Interprocess Communication"
section: "3.5.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#interprocess-communication"
---

### [3.5.4.2. Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#interprocess-communication)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#interprocess-communication "Permalink to this headline")

For very large computations, it is common to use multiple GPUs together to make use of more memory and more compute resources working together on a problem. Within a single system, or node in cluster computing terminology, multiple GPUs can be used in a single host process. This is described in [Section 3.4](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-gpu-introduction).

It is also common to use separate host processes spanning either a single computer or multiple computers. When multiple processes are working together, communication between them is known as interprocess communication. CUDA interprocess communication (CUDA IPC) provides mechanisms to share GPU buffers between different processes. [Section 4.15](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication) explains and demonstrates how CUDA IPC can be used to coordinate and communicate between different host processes.
