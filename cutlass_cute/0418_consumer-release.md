---
title: "Consumer release"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#consumer-release"
---

##### [Consumer release](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#consumer-release)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#consumer-release "Permalink to this headline")

The `consumer_release` method is to be used by consumer threads
to signal waiting producer threads that they have finished consuming data
associated with a particular stage of the pipeline.
This is a nonblocking instruction.
