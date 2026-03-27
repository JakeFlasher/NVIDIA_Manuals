---
title: "5.6.4.8.1. The Implicit (NULL) Stream"
section: "5.6.4.8.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#the-implicit-null-stream"
---

#### [5.6.4.8.1. The Implicit (NULL) Stream](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#the-implicit-null-stream)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#the-implicit-null-stream "Permalink to this headline")

Within a host program, the unnamed (NULL) stream has additional barrier synchronization semantics with other streams (see [Blocking and non-blocking streams and the default stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream) for details). The device runtime offers a single implicit, unnamed stream shared between all threads in a thread block, but as all named streams must be created with the `cudaStreamNonBlocking` flag, work launched into the NULL stream will not insert an implicit dependency on pending work in any other streams (including NULL streams of other thread blocks).
