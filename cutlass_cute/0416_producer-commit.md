---
title: "Producer commit"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#producer-commit"
---

##### [Producer commit](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#producer-commit)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#producer-commit "Permalink to this headline")

The `producer_commit` method is to be issued by asynchronous producer threads
after the instructions associated with a particular stage
(e.g., shared memory writes) have completed,
in order to notify the waiting asynchronous consumer threads.
This is a nonblocking instruction.

This API may result in a No-Op in some cases,
if the producer instructions also update the barrier stage associated automatically
(e.g., TMA_based producer threads using the  `PipelineTmaAsync ` class).
