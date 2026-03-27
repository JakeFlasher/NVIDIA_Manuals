---
title: "Producer acquire"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#producer-acquire"
---

##### [Producer acquire](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#producer-acquire)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#producer-acquire "Permalink to this headline")

The `producer_acquire` method is to be used by asynchronous producer threads
before issuing other instructions associated with a particular pipeline stage
(e.g., copy or write).

This is a blocking instruction
which blocks further execution of producer threads
unless the particular stage waiting to be acquired
is released by a consumer.

We say that a pipeline at its start is “empty” if producer threads are free to produce and do not need to wait for a consumer release – that is, if an acquire operation is expected to succeed.  If the pipeline at its start is empty, then we can either skip performing producer acquire operations during the first pass through the pipeline stages, or use the `make_producer_start_state` method.  The latter ensures that the acquire operation will succeed at the start of a pipeline.
