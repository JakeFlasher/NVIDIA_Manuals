---
title: "5.6.4.3. SM Id and Warp Id"
section: "5.6.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#sm-id-and-warp-id"
---

### [5.6.4.3. SM Id and Warp Id](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#sm-id-and-warp-id)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#sm-id-and-warp-id "Permalink to this headline")

Note that in PTX `%smid` and `%warpid` are defined as volatile values. The device runtime may reschedule thread blocks onto different SMs in order to more efficiently manage resources. As such, it is unsafe to rely upon `%smid` or `%warpid` remaining unchanged across the lifetime of a thread or thread block.
