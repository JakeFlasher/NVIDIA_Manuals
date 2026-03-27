---
title: "Consumer wait"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#consumer-wait"
---

##### [Consumer wait](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#consumer-wait)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#consumer-wait "Permalink to this headline")

The `consumer_wait` method is to be used by consumer threads
before consuming data from a particular pipeline stage
which is expected to be produced by producer threads.

This is a blocking instruction.  That is,
until the producer threads have committed to a particular stage,
this instruction is expected to block further execution of consumer threads.
