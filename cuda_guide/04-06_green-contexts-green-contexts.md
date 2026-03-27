---
title: "4.6. Green Contexts"
section: "4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html#green-contexts--green-contexts"
---

# [4.6. Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#green-contexts)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts "Permalink to this headline")

A green context (GC) is a lightweight context associated, from its creation, with a set of specific GPU resources.
Users can partition GPU resources, currently streaming multiprocessors (SMs) and work queues (WQs), during green context creation, so that GPU work targeting a green context can only use its provisioned SMs and work queues.
Doing so can be beneficial in reducing, or better controlling, interference due to use of common resources.
An application can have multiple green contexts.

Using green contexts does not require any GPU code (kernel) changes, just small host-side changes (e.g., green context creation and stream creation for this green context).
The green context functionality can be useful in various scenarios. For example, it can help ensure some SMs are always available for a latency-sensitive kernel to start executing, assuming no other
constraints, or provide a quick way to test the effect of using fewer SMs without any kernel modifications.

Green context support first became available via the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS).
Starting from CUDA 13.1, contexts are exposed in the CUDA runtime via the execution context (EC) abstraction.
Currently, an execution context can correspond to either the primary context (the context runtime API users have always implicitly interacted with) or a green context.
This section will use the terms _execution context_ and _green context_ interchangeably when referring to a green context.

With the runtime exposure of green contexts, using the CUDA runtime API directly is strongly recommended. This section will also solely use the CUDA runtime API.

The remaining of this section is organized as follows:
[Section 4.6.1](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-motivation) provides a motivating example, [Section 4.6.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-ease-of-use) highlights ease of use, and [Section 4.6.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-device-resource-and-desc) presents the device resource and resource descriptor structs.
[Section 4.6.4](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-creation-example)  explains how to create a green context, [Section 4.6.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-launching-work) how to launch work that targets it,
and [Section 4.6.6](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-apis) highlights some additional green context APIs. Finally, [Section 4.6.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#green-contexts-example) wraps up with an example.
